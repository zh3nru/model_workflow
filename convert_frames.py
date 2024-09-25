import os
import cv2
from pathlib import Path
from tqdm import tqdm
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

def convert_frames(video_path: Path, output_dir: Path, frames_per_second: int = 1):
    """
    Extracts frames from a video file and saves them to the specified directory.

    Args:
        video_path (Path): Path to the video file.
        output_dir (Path): Directory where extracted frames will be saved.
        frames_per_second (int): Number of frames to extract per second of video.
    """
    try:
        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        vidcap = cv2.VideoCapture(str(video_path))
        if not vidcap.isOpened():
            logging.error(f"Cannot open video file: {video_path}")
            return

        fps = vidcap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 25  # Default FPS if unable to get from video
            logging.warning(f"FPS not detected for {video_path}. Using default FPS={fps}")

        frame_interval = int(fps / frames_per_second)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = frame_count / fps
        total_frames_to_extract = frames_per_second * int(duration_seconds)

        logging.info(f"Processing video: {video_path}")
        logging.info(f"FPS: {fps}, Frame Interval: {frame_interval}, Total Frames to Extract: {total_frames_to_extract}")

        current_frame = 0
        extracted_frames = 0

        with tqdm(total=frame_count, desc=f"Extracting frames from {video_path.name}", unit="frame") as pbar:
            while True:
                success, frame = vidcap.read()
                if not success:
                    break

                if current_frame % frame_interval == 0:
                    frame_filename = f"{video_path.stem}_frame{extracted_frames + 1}.jpg"
                    frame_filepath = output_dir / frame_filename
                    
                    # Save the frame and check for success
                    if cv2.imwrite(str(frame_filepath), frame):
                        extracted_frames += 1
                    else:
                        logging.error(f"Failed to write frame {frame_filename} to {frame_filepath}")

                current_frame += 1
                pbar.update(1)

        vidcap.release()
        logging.info(f"Extracted {extracted_frames} frames from {video_path} to {output_dir}")

    except Exception as e:
        logging.error(f"Error processing video {video_path}: {e}")

def process_videos_in_parallel(video_files, frames_data_path, frames_per_second):
    """
    Processes multiple videos in parallel using ThreadPoolExecutor.

    Args:
        video_files (list): List of video files to process.
        frames_data_path (Path): Path to save frames.
        frames_per_second (int): Number of frames to extract per second.
    """
    with ThreadPoolExecutor() as executor:
        future_to_video = {
            executor.submit(convert_frames, video_file, frames_data_path / video_file.stem, frames_per_second): video_file
            for video_file in video_files
        }

        for future in tqdm(as_completed(future_to_video), total=len(video_files), desc="Processing videos", unit="video"):
            video_file = future_to_video[future]
            try:
                future.result()
            except Exception as exc:
                logging.error(f"Video {video_file} generated an exception: {exc}")

def convert_videos_to_frames(vids_data_dir: str = 'data/train_gen_vids', frames_data_dir: str = 'train_gen_frames', frames_per_second: int = 1):
    """
    Converts all videos in the vids_data_dir into image frames stored in frames_data_dir.

    Args:
        vids_data_dir (str): Directory containing videos organized by emotion.
        frames_data_dir (str): Directory where frames will be stored organized by emotion.
        frames_per_second (int): Number of frames to extract per second of video.
    """
    joint_data_path = Path(vids_data_dir)
    frames_data_path = Path(frames_data_dir)

    if not joint_data_path.exists():
        logging.critical(f"Video data directory does not exist: {joint_data_path}")
        sys.exit(1)

    frames_data_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Frames data directory set to: {frames_data_path}")

    # Iterate through each emotion directory
    for emotion_dir in joint_data_path.iterdir():
        if emotion_dir.is_dir():
            emotion_name = emotion_dir.name
            target_frames_dir = frames_data_path / emotion_name
            target_frames_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Processing emotion: {emotion_name}")
            logging.info(f"Frames will be saved to: {target_frames_dir}")

            # Collect all video files
            video_files = list(emotion_dir.glob('*'))
            video_files = [vf for vf in video_files if vf.is_file() and vf.suffix in ['.mp4', '.avi', '.mov', '.mkv']]

            if not video_files:
                logging.warning(f"No supported video files found in {emotion_dir}")
                continue

            process_videos_in_parallel(video_files, target_frames_dir, frames_per_second)

def setup_logging():
    """
    Sets up the logging configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

if __name__ == '__main__':
    setup_logging()

    # Retrieve environment variables or use default paths
    vids_dir = os.getenv('train_data_path', 'data/train_gen_vids')
    frames_dir = 'train_gen_frames'  # Fixed directory as per requirement
    frames_ps = 1  # Adjust as needed

    convert_videos_to_frames(vids_data_dir=vids_dir, frames_data_dir=frames_dir, frames_per_second=frames_ps)
