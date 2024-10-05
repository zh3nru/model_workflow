import os
import json
from supabase import create_client, Client
from pathlib import Path
from tqdm import tqdm
import requests
import logging
import sys
from urllib.parse import urlparse, unquote
from base64 import b64encode, b64decode

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("retrieve_data.log"),
        logging.StreamHandler()
    ]
)

GITHUB_API_URL = "https://api.github.com"
GITHUB_REPO = 'zh3nru/model_CI'
FLAG_FILE_PATH = 'flag.txt'

def get_github_file(repo_name, file_path, github_token):
    """
    Retrieves the content and SHA of a file from a GitHub repository.
    """
    url = f"{GITHUB_API_URL}/repos/{repo_name}/contents/{file_path}"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        file_info = response.json()
        content = b64decode(file_info['content']).decode('utf-8')
        sha = file_info['sha']
        logging.info(f"Successfully fetched {file_path} from {repo_name}.")
        return content, sha
    elif response.status_code == 404:
        logging.info(f"{file_path} does not exist in {repo_name}. It will be created.")
        return "", None
    else:
        logging.error(f"Failed to fetch {file_path} from GitHub. Status code: {response.status_code}. Response: {response.json()}")
        return None, None

def update_github_file(repo_name, file_path, content, github_token, commit_message="Update flag file", sha=None):
    """
    Updates or creates a file in a GitHub repository.
    """
    url = f"{GITHUB_API_URL}/repos/{repo_name}/contents/{file_path}"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    data = {
        "message": commit_message,
        "content": b64encode(content.encode('utf-8')).decode('utf-8'),
        "branch": "main"  # Adjust if you're using a different branch
    }
    
    if sha:
        data["sha"] = sha
    
    response = requests.put(url, headers=headers, data=json.dumps(data))
    
    if response.status_code in [200, 201]:
        action = "Updated" if sha else "Created"
        logging.info(f"{action} {file_path} in {repo_name} successfully.")
        return True
    else:
        logging.error(f"Failed to update {file_path} on GitHub. Status code: {response.status_code}. Response: {response.json()}")
        return False

def load_processed_ids(repo_name, github_token, file_path=FLAG_FILE_PATH):
    """
    Loads the list of processed IDs from the flag.txt file in the GitHub repository.
    """
    content, sha = get_github_file(repo_name, file_path, github_token)
    if content is None:
        logging.error("Unable to load processed IDs from GitHub.")
        sys.exit(1)
    processed_ids = set(line.strip() for line in content.splitlines() if line.strip())
    logging.info(f"Loaded {len(processed_ids)} processed IDs from {file_path} in {repo_name}.")
    return processed_ids, content, sha

def save_processed_ids(repo_name, github_token, new_ids, current_content, sha, file_path=FLAG_FILE_PATH):
    """
    Appends new processed IDs to the flag.txt file in the GitHub repository.
    """
    updated_content = current_content + ''.join(f"{record_id}\n" for record_id in new_ids)
    success = update_github_file(repo_name, file_path, updated_content, github_token, commit_message="Update processed IDs", sha=sha)
    if success:
        logging.info(f"Appended {len(new_ids)} new IDs to {file_path} in {repo_name}.")
    else:
        logging.error(f"Failed to append new IDs to {file_path} in {repo_name}.")

def upload_to_github(file_path, repo_name, github_token, target_folder, commit_message="Upload video file"):
    """
    Uploads a file to a specified folder in a GitHub repository.
    """
    github_api_url = f"{GITHUB_API_URL}/repos/{repo_name}/contents/{target_folder}/{file_path.name}"

    with open(file_path, "rb") as file:
        content = b64encode(file.read()).decode('utf-8')

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # Check if the file already exists to get its SHA
    get_response = requests.get(github_api_url, headers=headers)
    if get_response.status_code == 200:
        sha = get_response.json()['sha']
    elif get_response.status_code == 404:
        sha = None
    else:
        logging.error(f"Failed to check existence of {file_path.name} on GitHub. Status code: {get_response.status_code}. Response: {get_response.json()}")
        return

    data = {
        "message": commit_message,
        "content": content,
        "branch": "main"  # Adjust if you're using a different branch
    }

    if sha:
        data["sha"] = sha

    response = requests.put(github_api_url, headers=headers, data=json.dumps(data))

    if response.status_code in [201, 200]:
        logging.info(f"Successfully uploaded {file_path.name} to GitHub repository {repo_name}.")
    else:
        logging.error(f"Failed to upload {file_path.name} to GitHub. Status code: {response.status_code}. Response: {response.json()}")

def retrieve_data(supabase: Client, table_name: str = 'videos_data', data_dir: str = 'data/train_gen_vids',
                 repo_name: str = '', target_folder: str = '', github_token: str = '',
                 processed_ids_file: str = FLAG_FILE_PATH):
    """
    Retrieves new data from Supabase, downloads videos, organizes them, uploads to GitHub,
    and records processed IDs to avoid reprocessing in future runs.
    
    Sets an output variable 'NEW_DATA_PROCESSED' to 'true' if any new data was processed.
    """
    # Load already processed IDs from GitHub
    processed_ids, current_flag_content, flag_sha = load_processed_ids(repo_name, github_token, processed_ids_file)

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
    logging.info(f"Found {len(new_records)} new data to process out of {len(data)} total records.")

    if not new_records:
        logging.info("No new data to process.")
        return False  # No new data processed

    newly_processed_ids = set()

    for record in tqdm(new_records, desc="Retrieving Data"):
        record_id = record.get('id')
        if record_id is None:
            logging.warning(f"Skipping data with missing ID: {record}")
            continue

        video_url = record.get('video_path')
        emotion = record.get('emotion_class')

        if not video_url or not emotion:
            logging.warning(f"Skipping data with missing data: {record}")
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

        # Removed the existence check since IDs are unique
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
 
        # After successful processing (download and upload), add the record ID to the set
        newly_processed_ids.add(str(record_id))

    if newly_processed_ids:
        # Update flag.txt on GitHub with new IDs
        save_processed_ids(repo_name, github_token, newly_processed_ids, current_flag_content, flag_sha, processed_ids_file)
        return True
    else:
        return False

if __name__ == '__main__':
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    MY_TOKEN = os.getenv('MY_TOKEN')
    GITHUB_REPO = 'zh3nru/model_CI'
    TARGET_FOLDER = 'data/train_gen_vids'
    PROCESSED_IDS_FILE = FLAG_FILE_PATH

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
