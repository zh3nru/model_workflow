import os
from supabase import create_client, Client
import logging
import sys

def setup_logging():
    # logging
    logging.basicConfig(
        level=logging.INFO,  
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_supabase_client() -> Client:
    
    # Retrieves Supabase credentials from environment variables and creates a Supabase client.
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')

    if not SUPABASE_URL or not SUPABASE_KEY:
        logging.critical("Supabase credentials not found in environment variables. No 'SUPABASE_URL' and 'SUPABASE_KEY'.")
        raise EnvironmentError("Supabase credentials not found in environment variables.")

    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logging.info("Supabase client created successfully.")
        return supabase
    except Exception as e:
        logging.critical(f"Failed to create Supabase client: {e}")
        raise

def check_data(supabase: Client, table_name: str = 'videos_data'):
    
    # Checks if there is any data present in the specified columns of the table.
    logging.info(f"Checking for data in table '{table_name}' for 'emotion_class' and 'video_path' columns")

    try:
        # Check 'emotion_class' column for data
        emotion_response = (
            supabase.table(table_name)
            .select('id')
            .filter('emotion_class', 'neq', None)
            .filter('emotion_class', 'neq', '')
            .limit(1)
            .execute()
        )
        emotion_exists = len(emotion_response.data) > 0
        logging.info(f"Data exists in 'emotion_class' column: {emotion_exists}")

        # Check 'video_path' column for data
        video_response = (
            supabase.table(table_name)
            .select('id')
            .filter('video_path', 'neq', None)
            .filter('video_path', 'neq', '')
            .limit(1)
            .execute()
        )
        video_exists = len(video_response.data) > 0
        logging.info(f"Data exists in 'video_path' column: {video_exists}")

        data_present = emotion_exists and video_exists
        logging.info(f"Overall data (both columns have data): {data_present}")

        # Set GitHub Actions output
        with open(os.environ['GITHUB_OUTPUT'], 'a') as gh_output:
            gh_output.write(f'DATA_PRESENT={str(data_present).lower()}\n')

    except Exception as e:
        logging.error(f"An error occurred while querying Supabase: {e}")
        raise


def main():
    setup_logging()
    logging.info("Starting the data check process.")

    try:
        # Create a Supabase client
        supabase = get_supabase_client()
    except Exception as e:
        logging.critical(f"Initialization failed: {e}")
        sys.exit(1)

    try:
        check_data(supabase, table_name='videos_data')  
    except Exception as e:
        logging.critical(f"Failed to check data: {e}")
        sys.exit(1)

    logging.info("Data check process completed successfully.")

if __name__ == '__main__':
    main()
