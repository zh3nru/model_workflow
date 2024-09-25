import os
import cv2
import requests
import base64
from pathlib import Path
from tqdm import tqdm
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Constants
GITHUB_API_URL = "https://api.github.com"
GITHUB_REPO = "zh3nru/model_CI"  # Replace with your repository in "username/repo" format
GITHUB_BRANCH = "main"  # Replace with your target branch name
MY_TOKEN = os.getenv('MY_TOKEN')  # Ensure you set this environment variable

def upload_to_github(file_path, github_path):
    try:
        if not MY_TOKEN:
            logging.error("GitHub token is not set. Please set the MY_TOKEN environment variable.")
            return False

        with open(file_path, 'rb') as f:
            content = f.read()

        encoded_content = base64.b64encode(content).decode('utf-8')

        # Adjust the API URL to reflect the correct path
        api_url = f"{GITHUB_API_URL}/repos/{GITHUB_REPO}/contents/{github_path}"
        headers = {
            "Authorization": f"token {MY_TOKEN}",
            "Content-Type": "application/json",
        }

        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            sha = response.json()["sha"]
        else:
            sha = None

        data = {
            "message": "Upload frame",
            "content": encoded_content,
            "branch": GITHUB_BRANCH,
        }
        if sha:
            data["sha"] = sha

        response = requests.put(api_url, json=data, headers=headers)

        if response.status_code in [200, 201]:
            logging.info(f"Successfully uploaded {file_path} to GitHub at {github_path}")
            return True
        else:
            logging.error(f"Failed to upload {file_path} to GitHub: {response.json()}")
            return False

    except Exception as e:
        logging.error(f"Error uploading {file_path} to GitHub: {e}")
        return False

def convert_frames(video_path: Path, output_dir: Path, frames_per_second: int = 1):
    try:
        output_dir.mkdir(parents=True, exist_ok=True) 

        vidcap = cv2.VideoCapture(str(video_path))
        if not vidcap.isOpened():
            logging.error(f"Cannot open video file: {video_path}")
            return

        fps = vidcap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 25  
            logging.warning(f"FPS not detected for {video_path}. Using default FPS={fps}")

        frame_interval = int(fps / frames_per_second)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        logging.info(f"Processing video: {video_path}")
        logging.info(f"FPS: {fps}, Frame Interval: {frame_interval}")

        current_frame = 0
        extracted_frames = 0

        with tqdm(total=frame_count, desc=f"Extracting frames from {video_path.name}", unit="frame") as pbar:
            while True:
                success, frame = vidcap.read()
                if not success:
                    logging.info(f"Finished reading video file or encountered an error at frame {current_frame}.")
                    break

                if current_frame % frame_interval == 0:
                    frame_filename = f"{video_path.stem}_frame{extracted_frames + 1}.jpg"
                    frame_filepath = output_dir / frame_filename

                    if cv2.imwrite(str(frame_filepath), frame):
                        extracted_frames += 1

                        # Construct the github_path to match "data/train_gen_frames/Sadness/video_1.jpg"
                        github_path = f"data/train_gen_frames/{output_directory.name}/{frame_filename}"
                        
                        # Upload frame to GitHub using the full path
                        if not upload_to_github(frame_filepath, github_path):
                            logging.error(f"Failed to upload frame {frame_filename} to GitHub.")
                    else:
                        logging.error(f"Failed to write frame {frame_filename} to {frame_filepath}")

                current_frame += 1
                pbar.update(1)

        vidcap.release()
        logging.info(f"Extracted {extracted_frames} frames from {video_path} to {output_dir}")

    except Exception as e:
        logging.error(f"Error processing video {video_path}: {e}")

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,  
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

if __name__ == '__main__':
    setup_logging()

    vids_dir = 'data/train_gen_vids'
    frames_dir = 'data/train_gen_frames'
    frames_ps = 1  

    joint_data_path = Path(vids_dir)
    frames_data_path = Path(frames_dir)

    if not joint_data_path.exists():
        logging.critical(f"Video data directory does not exist: {joint_data_path}")
        sys.exit(1)

    frames_data_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Frames data directory set to: {frames_data_path}")

    # Modified part: Scan each subfolder in the joint_data_path
    for emotion_subfolder in joint_data_path.iterdir():
        if emotion_subfolder.is_dir():
            emotion_name = emotion_subfolder.name
            for video_file in emotion_subfolder.glob('*.mp4'):  
                output_directory = frames_data_path / emotion_name  # Ensure frames are saved in the emotion-specific folder
                output_directory.mkdir(parents=True, exist_ok=True)
                convert_frames(video_file, output_directory, frames_per_second=frames_ps)
