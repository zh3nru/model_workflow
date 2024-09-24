import os
import json
from supabase import create_client, Client
from pathlib import Path
from tqdm import tqdm
import requests
import logging
import sys
from urllib.parse import urlparse, unquote
from base64 import b64encode

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("retrieve_data.log"),  # Log to a file
        logging.StreamHandler()                   # Log to console
    ]
)

def upload_to_github(file_path, repo_name, github_token, target_folder, commit_message="Upload video file"):
    """
    Uploads a file to a specified folder in a GitHub repository.

    Args:
        file_path (Path): The path to the local file.
        repo_name (str): The name of the GitHub repository (e.g., "username/repo").
        github_token (str): Personal access token for GitHub.
        target_folder (str): The folder in the GitHub repo where the file will be uploaded.
        commit_message (str): Commit message for the upload.
    """
    github_api_url = f"https://api.github.com/repos/{repo_name}/contents/{target_folder}/{file_path.name}"

    with open(file_path, "rb") as file:
        content = b64encode(file.read()).decode('utf-8')

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Content-Type": "application/json"
    }

    data = json.dumps({
        "message": commit_message,
        "content": content
    })

    response = requests.put(github_api_url, headers=headers, data=data)

    if response.status_code == 201:
        logging.info(f"Successfully uploaded {file_path.name} to GitHub repository {repo_name}.")
    else:
        logging.error(f"Failed to upload {file_path.name} to GitHub. Status code: {response.status_code}. Response: {response.json()}")

def retrieve_data(supabase: Client, table_name: str = 'videos_data', data_dir: str = 'data/train_gen_vids', repo_name: str = '', target_folder: str = '', github_token: str = ''):
    """
    Retrieves new data using URLs in the table, saves them locally in folders based on emotion, and uploads them to a GitHub repository.

    Args:
        supabase (Client): Supabase client instance.
        table_name (str): Name of the table to query.
        data_dir (str): Base directory to store training videos.
        repo_name (str): GitHub repository name (e.g., "username/repo").
        target_folder (str): Folder path in the GitHub repo.
        github_token (str): Personal access token for GitHub.
    """
    # Fetch all records to process
    response = supabase.table(table_name).select('*').execute()

    if not response or not hasattr(response, 'data') or response.data is None:
        logging.error(f"Error retrieving data from the table. Response received: {response}")
        return

    data = response.data

    if not data:
        logging.info("No data retrieved from the database.")
        return

    for record in tqdm(data, desc="Retrieving Data"):
        video_url = record.get('video_path')
        emotion = record.get('emotion_class')

        if not video_url or not emotion:
            logging.warning(f"Skipping record with missing data: {record}")
            continue

        # Normalize the emotion to be used as a directory name
        emotion_normalized = emotion.strip().replace(" ", "_")
        emotion_dir = Path(data_dir) / emotion_normalized

        try:
            emotion_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Directory {emotion_dir} is ready.")
        except Exception as e:
            logging.error(f"Failed to create directory {emotion_dir}: {e}")
            continue

        parsed_url = urlparse(video_url)
        video_filename = unquote(Path(parsed_url.path).name)
        video_full_path = emotion_dir / video_filename

        if video_full_path.exists():
            logging.info(f"Video already exists: {video_full_path}. Skipping download.")
            continue

        try:
            logging.info(f"Downloading video from URL: {video_url}")
            response = requests.get(video_url, stream=True, timeout=30)

            if response.status_code == 200:
                with open(video_full_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                logging.info(f"Successfully downloaded video: {video_full_path}")

                if video_full_path.exists() and video_full_path.stat().st_size > 0:
                    logging.info(f"File saved correctly: {video_full_path}, Size: {video_full_path.stat().st_size} bytes")

                    # Upload to GitHub with the updated path
                    upload_to_github(video_full_path, repo_name, github_token, f"{target_folder}/{emotion_normalized}")
                else:
                    logging.error(f"File was not saved correctly or is empty: {video_full_path}")
            else:
                logging.error(f"Failed to download video from URL: {video_url}. Status code: {response.status_code}")
                continue

        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading video from URL {video_url}: {e}")
            continue
        except Exception as e:
            logging.error(f"Unexpected error occurred while saving video: {e}")
            continue

if __name__ == '__main__':
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    MY_TOKEN = os.getenv('MY_TOKEN')
    GITHUB_REPO = 'zh3nru/model_CI'  # Replace with your GitHub repo name
    TARGET_FOLDER = 'data/train_gen_vids'  # The folder path in your GitHub repo where files will be uploaded

    if not SUPABASE_URL or not SUPABASE_KEY or not MY_TOKEN:
        logging.critical("Supabase credentials or GitHub token not found in environment variables.")
        sys.exit(1)

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    retrieve_data(
        supabase,
        table_name='videos_data',
        data_dir='data/train_gen_vids',
        repo_name=GITHUB_REPO,
        target_folder=TARGET_FOLDER,
        github_token=MY_TOKEN
    )
