import imageio
import os

def extract_images_from_video(mp4_path, output_folder, fps=30):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    video_reader = imageio.get_reader(mp4_path, 'ffmpeg')

    frame_count = 10
    for frame in video_reader:
        # Save the frame as an image
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        imageio.imwrite(frame_filename, frame)
        frame_count += 1

    video_reader.close()
    print(f"Extracted {frame_count} frames to {output_folder}")

# Example usage
mp4_path = '/home/shellyf/Documents/research_results/rollouts/2025_01_19/2025_01_19-14_28_46--episode=3--success=True--task=put_both_the_alphabet_soup_and_the_tomato_sauce_in.mp4'
output_folder = '/home/shellyf/Documents/research_results/rollouts/2025_01_19/2025_01_19-14_28_46--episode=3--success=True--task=put_both_the_alphabet_soup_and_the_tomato_sauce_in'
extract_images_from_video(mp4_path, output_folder, fps=30)