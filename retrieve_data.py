import os
from supabase import create_client, Client
from pathlib import Path
from tqdm import tqdm
import requests

def retrieve_data(supabase: Client, table_name: str = 'videos_data', storage_bucket: str = 'videos_bucket', data_dir: str = 'data/train_gen_vids'):
    """
    Retrieves new data from Supabase and organizes it into emotion-specific training directories.

    Args:
        supabase (Client): Supabase client instance.
        table_name (str): Name of the table to query.
        storage_bucket (str): Name of the storage bucket containing videos.
        data_dir (str): Base directory to store training videos.
    """
    # Fetch all records to process
    response = supabase.table(table_name).select('*').execute()

    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code}")

    data = response.data

    # Iterate through records and download videos
    for record in tqdm(data, desc="Retrieving Data"):
        video_path = record.get('video_path')  
        emotion = record.get('emotion_class')        

        if not video_path or not emotion:
            print(f"Skipping record with missing data: {record}")
            continue

        # Normalize emotion label to lowercase for directory naming consistency
        emotion_normalized = emotion

        # Define the target directory based on emotion
        emotion_dir = Path(data_dir) / emotion_normalized
        emotion_dir.mkdir(parents=True, exist_ok=True)

        # Define the video filename
        video_filename = Path(video_path).name  # Extracts the filename from the URL/path
        video_full_path = emotion_dir / video_filename

        # Skip downloading if the video already exists
        if video_full_path.exists():
            print(f"Video already exists: {video_full_path}. Skipping download.")
            continue

        try:
            # If video_path is a full URL, use requests to download
            if video_path.startswith('http://') or video_path.startswith('https://'):
                response = requests.get(video_path, stream=True)
                if response.status_code == 200:
                    with open(video_full_path, 'wb') as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                else:
                    print(f"Failed to download video: {video_path}. Status Code: {response.status_code}")
                    continue
            else:
                # If video_path is a path in Supabase Storage, download using Supabase Storage API
                storage = supabase.storage()
                # Assuming video_path is the path within the bucket
                file_response = storage.from_(storage_bucket).download(video_path)
                if file_response.status_code == 200:
                    with open(video_full_path, 'wb') as f:
                        f.write(file_response.data)
                else:
                    print(f"Failed to download video from storage: {video_path}. Status Code: {file_response.status_code}")
                    continue

            print(f"Downloaded video: {video_full_path}")

        except Exception as e:
            print(f"Error downloading video {video_path}: {e}")
            continue

if __name__ == '__main__':
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise EnvironmentError("Supabase credentials not found in environment variables.")

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    retrieve_data(
        supabase,
        table_name='videos_data',          
        storage_bucket='videos_bucket',     
        data_dir='data/train_gen_vids'    
    )
