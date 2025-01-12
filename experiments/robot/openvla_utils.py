"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time

import numpy as np
import tensorflow as tf
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register("openvla", OpenVLAConfig, OpenVLAForActionPrediction)

    vla = OpenVLAForActionPrediction.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Move model to device.
    # Note: `.to()` is not supported for 8-bit or 4-bit bitsandbytes models, but the model will
    #       already be set to the right devices and casted to the correct dtype upon loading.
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        if not torch.cuda.is_available() and cfg.device == torch.device("cuda:0"):
            print("WARNING: CUDA is not available. Moving model to CPU.")
            cfg.device = torch.device("cpu")
        vla = vla.to(cfg.device)
        # vla = vla.to(DEVICE)

    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join("experiments/robot/libero/experiments", "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, device, center_crop=False):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process inputs.
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

    # Get action and probabilities for each action
    action, probs = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action, probs

def add_text_to_image(image: np.ndarray, text: str, position: tuple = (5, 5)) -> np.ndarray:
    # Convert the NumPy array image to a PIL image
    pil_image = Image.fromarray(image)

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Load a font
    font = ImageFont.truetype(os.path.join(os.getcwd(), r'Arial.ttf'), 9)

    # Draw the text on the image
    draw.text(position, text, font=font, fill=(255, 255, 255),align ="left")

    # Convert the PIL image back to a NumPy array
    return np.array(pil_image)

def probs_for_calibration(probs, type="mul"):
    """
    applying the type function on the probabilities list
    Options: mul, max, min, avg, perplexity,
    """
    if type == "mul": # multiply all probabilities in the array by each other
        return np.prod(probs)
    elif type == "max": # return the maximum probability in the array
        return np.max(probs)
    elif type == "min": # return the minimum probability in the array
        return np.min(probs)
    elif type == "avg": # return the average probability in the array
        return np.mean(probs)
    # elif type == "perplexity": # calculate the perplexity of the probabilities
    #     return 2 ** (-np.mean(np.log2(probs)))
    else:
        raise ValueError("type should be either 'mul' or 'div'")


def calculate_ece(confidences, success_rate, num_bins=10):
    """
    Computes the Expected Calibration Error (ECE).
    Args:
        confidences (list): List of confidence scores corresponding to predictions (0 to 1).
        success_rate (list): List of ground truth labels.
        num_bins (int): Number of bins for bucketing predictions by confidence.
    Returns:
        float: Expected Calibration Error (ECE).
    """

    if len(success_rate) != len(confidences) :
        raise ValueError("Length of confidences, and ground_truth must be the same.")
    # Initialize bins
    bin_boundaries = np.linspace(min(confidences), max(confidences), num_bins)
    bin_indices = np.digitize(confidences, bin_boundaries, right=False) - 1 # Adjust index to 0-based
    ece = 0.0
    empty_bins = 0
      # Iterate through each bin
    for bin_idx in range(num_bins):
        # Get indices of predictions in the current bin
        bin_mask = bin_indices == bin_idx
        if np.sum(bin_mask) == 0:
            empty_bins += 1
            continue  # Skip empty bins
        # Get accuracy and average confidence for the bin
        bin_confidences = np.array(confidences)[bin_mask]
        bin_success_rate = np.array(success_rate)[bin_mask]
        bin_avg_confidence = np.mean(bin_confidences)
        bin_avg_success_rate = np.mean(bin_success_rate)

        # Compute contribution to ECE
        ece += np.abs(bin_avg_confidence - bin_avg_success_rate)

    try:
        returned_ece = ece / (num_bins - empty_bins)
    except ZeroDivisionError:
        returned_ece = 1.0
    return returned_ece