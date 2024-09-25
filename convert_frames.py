import os
import cv2
from pathlib import Path
from tqdm import tqdm
import logging
import sys

def convert_frames(video_path: Path, output_dir: Path, frames_per_second: int = 1, max_frames: int = 5):
    """
    Extracts frames from a video file and saves them to the specified directory, up to a maximum of 5 frames.

    Args:
        video_path (Path): Path to the video file.
        output_dir (Path): Directory where extracted frames will be saved.
        frames_per_second (int): Number of frames to extract per second of video.
        max_frames (int): Maximum number of frames to extract from the video.
    """
    try:
        output_dir = output_dir.resolve()  # Ensure absolute path
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created output directory: {output_dir}")

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
        total_frames_to_extract = min(frames_per_second * int(duration_seconds), max_frames)

        logging.info(f"Processing video: {video_path}")
        logging.info(f"FPS: {fps}, Frame Interval: {frame_interval}, Total Frames to Extract: {total_frames_to_extract}")

        current_frame = 0
        extracted_frames = 0

        while extracted_frames < max_frames:
            success, frame = vidcap.read()
            if not success:
                logging.error(f"Failed to read frame at position {current_frame}")
                break

            if frame is None:
                logging.error(f"Frame is None at position {current_frame}. Skipping.")
                continue

            if current_frame % frame_interval == 0:
                frame_filename = f"{video_path.stem}_frame{extracted_frames + 1}.jpg"
                frame_filepath = output_dir / frame_filename
                
                logging.info(f"Attempting to save frame {extracted_frames + 1} to {frame_filepath}")

                # Attempt to save the frame
                if cv2.imwrite(str(frame_filepath), frame):
                    logging.info(f"Saved frame {extracted_frames + 1} as {frame_filename}")
                    
                    # Verify if the image has been saved
                    if frame_filepath.exists():
                        img_test = cv2.imread(str(frame_filepath))
                        if img_test is not None:
                            logging.info(f"Verified saved frame {extracted_frames + 1} at {frame_filepath}")
                        else:
                            logging.error(f"Unable to read back saved frame {extracted_frames + 1} from {frame_filepath}")
                    else:
                        logging.error(f"Frame file {frame_filepath} does not exist after attempting to save.")
                    
                    extracted_frames += 1
                else:
                    logging.error(f"Failed to save frame {extracted_frames + 1} as {frame_filename}")

                if extracted_frames >= max_frames:
                    break

            current_frame += 1

        vidcap.release()
        logging.info(f"Extracted {extracted_frames} frames from {video_path}")

    except Exception as e:
        logging.error(f"Error processing video {video_path}: {e}")

def convert_videos_to_frames(vids_data_dir: str = 'data/train_gen_vids', frames_data_dir: str = 'data/train_gen_frames', frames_per_second: int = 1, max_frames: int = 5):
    """
    Converts all videos in the vids_data_dir into image frames stored in frames_data_dir.

    Args:
        vids_data_dir (str): Directory containing videos organized by emotion.
        frames_data_dir (str): Directory where frames will be stored organized by emotion.
        frames_per_second (int): Number of frames to extract per second of video.
        max_frames (int): Maximum number of frames to extract per video.
    """
    joint_data_path = Path(vids_data_dir).resolve()
    frames_data_path = Path(frames_data_dir).resolve()

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

            # Iterate through each video file in the emotion directory
            video_files = list(emotion_dir.glob('*'))
            for video_file in tqdm(video_files, desc=f"Processing {emotion_name}", unit="video"):
                if video_file.is_file() and video_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    # Check if frames already extracted for this video
                    if any(target_frames_dir.glob(f"{video_file.stem}_frame*.jpg")):
                        logging.info(f"Frames already exist for video {video_file.name}. Skipping extraction.")
                        continue

                    # Save frames directly in the emotion directory
                    convert_frames(video_file, target_frames_dir, frames_per_second=frames_per_second, max_frames=max_frames)
                else:
                    logging.warning(f"Unsupported or non-file item skipped: {video_file}")

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

    # Retrieve environment variables or use absolute paths
    vids_dir = os.getenv('train_data_path', os.path.abspath('data/train_gen_vids'))
    frames_dir = os.path.abspath('data/train_gen_frames')  # Using absolute path
    frames_ps = 1  # Adjust as needed
    max_frames = 5  # Extract up to 5 frames per video

    convert_videos_to_frames(vids_data_dir=vids_dir, frames_data_dir=frames_dir, frames_per_second=frames_ps, max_frames=max_frames)
