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
        storage_bucket (str): Name of the storage bucket containing images.
        data_dir (str): Base directory to store training images.
    """
    # Fetch all records to process (assuming check_new_data.py has already updated flag)
    response = supabase.table(table_name).select('*').execute()

    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.status_code}")

    data = response.data

    # Iterate through records and download images
    for record in tqdm(data, desc="Retrieving Data"):
        image_url = record.get('image_url')  # Adjust field name as necessary
        emotion = record.get('emotion')      # Adjust field name as necessary

        if not image_url or not emotion:
            print(f"Skipping record with missing data: {record}")
            continue

        # Normalize emotion label to lowercase for directory naming consistency
        emotion_normalized = emotion

        # Define the target directory based on emotion
        emotion_dir = Path(data_dir) / emotion_normalized
        emotion_dir.mkdir(parents=True, exist_ok=True)

        # Define the image filename
        image_filename = Path(image_url).name  # Extracts the filename from the URL/path
        image_path = emotion_dir / image_filename

        # Skip downloading if the image already exists
        if image_path.exists():
            print(f"Image already exists: {image_path}. Skipping download.")
            continue

        try:
            # If image_url is a full URL, use requests to download
            if image_url.startswith('http://') or image_url.startswith('https://'):
                response = requests.get(image_url, stream=True)
                if response.status_code == 200:
                    with open(image_path, 'wb') as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                else:
                    print(f"Failed to download image: {image_url}. Status Code: {response.status_code}")
                    continue
            else:
                # If image_url is a path in Supabase Storage, download using Supabase Storage API
                storage = supabase.storage()
                # Assuming image_url is the path within the bucket
                file_response = storage.from_(storage_bucket).download(image_url)
                if file_response.status_code == 200:
                    with open(image_path, 'wb') as f:
                        f.write(file_response.data)
                else:
                    print(f"Failed to download image from storage: {image_url}. Status Code: {file_response.status_code}")
                    continue

            print(f"Downloaded image: {image_path}")

        except Exception as e:
            print(f"Error downloading image {image_url}: {e}")
            continue

if __name__ == '__main__':
    SUPABASE_URL = os.getenv('https://zpnrhnnbetfdvnffcrmj.supabase.co')
    SUPABASE_KEY = os.getenv('eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpwbnJobm5iZXRmZHZuZmZjcm1qIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcyMTYxNTExNywiZXhwIjoyMDM3MTkxMTE3fQ.mAt9GY5mJgn5nhEPGPIP31uiJWNNTVG3eEvv2w9smdk')

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise EnvironmentError("Supabase credentials not found in environment variables.")

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    retrieve_data(
        supabase,
        table_name='videos_data',           # Table name is 'videos_data'
        storage_bucket='videos_bucket',     # Replace 'images_bucket' with your actual storage bucket name
        data_dir='data/train_gen_vids'    # Ensure this matches the path used in train.py
    )
