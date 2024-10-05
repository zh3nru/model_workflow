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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("retrieve_data.log"),
        logging.StreamHandler()
    ]
)

def upload_to_github(file_path, repo_name, github_token, target_folder, commit_message="Upload video file"):
    """
    Uploads a file to a specified folder in a GitHub repository.
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

    if response.status_code in [201, 200]:
        logging.info(f"Successfully uploaded {file_path.name} to GitHub repository {repo_name}.")
    else:
        logging.error(f"Failed to upload {file_path.name} to GitHub. Status code: {response.status_code}. Response: {response.json()}")

def load_processed_ids(file_path):
    """
    Loads the list of processed IDs from a text file.
    """
    if not os.path.exists(file_path):
        logging.info(f"Processed IDs file {file_path} does not exist. A new one will be created.")
        return set()

    with open(file_path, 'r') as f:
        processed_ids = set(line.strip() for line in f if line.strip())
    logging.info(f"Loaded {len(processed_ids)} processed IDs from {file_path}.")
    return processed_ids

def save_processed_id(file_path, record_id):
    """
    Appends a processed ID to the text file.
    """
    with open(file_path, 'a') as f:
        f.write(f"{record_id}\n")
    logging.debug(f"Saved processed ID {record_id} to {file_path}.")

def retrieve_data(supabase: Client, table_name: str = 'videos_data', data_dir: str = 'data/train_gen_vids',
                 repo_name: str = '', target_folder: str = '', github_token: str = '',
                 processed_ids_file: str = 'flag.txt'):
    """
    Retrieves new data from Supabase, downloads videos, organizes them, uploads to GitHub,
    and records processed IDs to avoid reprocessing in future runs.
    
    Sets an output variable 'NEW_DATA_PROCESSED' to 'true' if any new data was processed.
    """
    # Load already processed IDs
    processed_ids = load_processed_ids(processed_ids_file)

    # Fetch data from Supabase
    response = supabase.table(table_name).select('*').execute()

    if not response or not hasattr(response, 'data') or response.data is None:
        logging.error(f"Error retrieving data from the table. Response received: {response}")
        return False  # No new data processed

    data = response.data

    if not data:
        logging.info("No data retrieved from the database.")
        return False  # No new data processed

    new_records = [record for record in data if str(record.get('id')) not in processed_ids]
    logging.info(f"Found {len(new_records)} new records to process out of {len(data)} total records.")

    if not new_records:
        logging.info("No new records to process.")
        return False  # No new data processed

    for record in tqdm(new_records, desc="Retrieving Data"):
        record_id = record.get('id')
        if record_id is None:
            logging.warning(f"Skipping record with missing ID: {record}")
            continue

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
        else:
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
                        # Optionally, you might want to remove the empty file
                        video_full_path.unlink(missing_ok=True)
                        continue
                else:
                    logging.error(f"Failed to download video from URL: {video_url}. Status code: {response.status_code}")
                    continue

            except requests.exceptions.RequestException as e:
                logging.error(f"Error downloading video from URL {video_url}: {e}")
                continue
            except Exception as e:
                logging.error(f"Unexpected error occurred while saving video: {e}")
                continue

        # After successful processing (download and upload), save the record ID
        save_processed_id(processed_ids_file, record_id)

    # If the function reaches here, new data was processed
    return True

if __name__ == '__main__':
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    MY_TOKEN = os.getenv('MY_TOKEN')
    GITHUB_REPO = 'zh3nru/model_CI'
    TARGET_FOLDER = 'data/train_gen_vids'
    PROCESSED_IDS_FILE = 'flag.txt'  

    if not SUPABASE_URL or not SUPABASE_KEY or not MY_TOKEN:
        logging.critical("Supabase credentials or GitHub token not found in environment variables.")
        sys.exit(1)

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Run the data retrieval process
    new_data_processed = retrieve_data(
        supabase,
        table_name='videos_data',
        data_dir='data/train_gen_vids',
        repo_name=GITHUB_REPO,
        target_folder=TARGET_FOLDER,
        github_token=MY_TOKEN,
        processed_ids_file=PROCESSED_IDS_FILE
    )

    # Set GitHub Actions output using the new method
    if 'GITHUB_OUTPUT' in os.environ:
        with open(os.environ['GITHUB_OUTPUT'], 'a') as gh_output:
            gh_output.write(f'NEW_DATA_PROCESSED={str(new_data_processed).lower()}\n')
    else:
        logging.warning("GITHUB_OUTPUT environment variable not found. Cannot set output variable.")
