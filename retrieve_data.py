import os
from supabase import create_client, Client
from pathlib import Path
from tqdm import tqdm
import requests
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("retrieve_data.log"),  # Log to a file
        logging.StreamHandler()                   # Log to console
    ]
)

def retrieve_data(supabase: Client, table_name: str = 'videos_data', storage_bucket: str = 'videos_bucket', data_dir: str = 'data/train_gen_vids'):
    """
    Retrieves new data from Supabase Storage using URLs in the table and organizes it into emotion-specific training directories.

    Args:
        supabase (Client): Supabase client instance.
        table_name (str): Name of the table to query.
        storage_bucket (str): Name of the storage bucket containing videos.
        data_dir (str): Base directory to store training videos.
    """
    # Fetch all records to process
    response = supabase.table(table_name).select('*').execute()

    if response.error:
        logging.error(f"Error retrieving data from the table: {response.error}")
        return

    data = response.data

    if not data:
        logging.info("No data retrieved from the database.")
        return

    # Iterate through records and download videos
    for record in tqdm(data, desc="Retrieving Data"):
        video_path = record.get('video_path')  # Assumes video_path contains the path to the video in the bucket
        emotion = record.get('emotion_class')  # Adjust field name as necessary

        if not video_path or not emotion:
            logging.warning(f"Skipping record with missing data: {record}")
            continue

        # Normalize emotion label to lowercase for directory naming consistency
        emotion_normalized = emotion.lower()

        # Define the target directory based on emotion
        emotion_dir = Path(data_dir) / emotion_normalized
        emotion_dir.mkdir(parents=True, exist_ok=True)

        # Define the video filename
        video_filename = Path(video_path).name  # Extracts the filename from the path
        video_full_path = emotion_dir / video_filename

        # Skip downloading if the video already exists
        if video_full_path.exists():
            logging.info(f"Video already exists: {video_full_path}. Skipping download.")
            continue

        try:
            # Use Supabase storage client to download the file
            storage = supabase.storage()
            file_response = storage.from_(storage_bucket).download(video_path)
            
            if file_response.data:
                with open(video_full_path, 'wb') as f:
                    f.write(file_response.data)
                logging.info(f"Successfully downloaded video: {video_full_path}")
            else:
                logging.error(f"Failed to download video from storage: {video_path}.")
                continue

        except Exception as e:
            logging.exception(f"Error downloading video {video_path}: {e}")
            continue


if __name__ == '__main__':
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')

    if not SUPABASE_URL or not SUPABASE_KEY:
        logging.critical("Supabase credentials not found in environment variables.")
        sys.exit(1)

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    retrieve_data(
        supabase,
        table_name='videos_data',          
        storage_bucket='videos_bucket',     
        data_dir='data/train_gen_vids'    
    )
