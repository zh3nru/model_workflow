import os
from supabase import create_client, Client
import logging
import sys

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

def get_supabase_client() -> Client:
    """
    Retrieves Supabase credentials from environment variables and creates a Supabase client.
    
    Raises:
        EnvironmentError: If SUPABASE_URL or SUPABASE_KEY is not set.
        Exception: If creating the Supabase client fails.
    
    Returns:
        Client: An instance of the Supabase client.
    """
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')

    if not SUPABASE_URL or not SUPABASE_KEY:
        logging.critical("Supabase credentials not found in environment variables. Please set 'SUPABASE_URL' and 'SUPABASE_KEY'.")
        raise EnvironmentError("Supabase credentials not found in environment variables.")

    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logging.info("Supabase client created successfully.")
        return supabase
    except Exception as e:
        logging.critical(f"Failed to create Supabase client: {e}")
        raise

def check_data_presence(supabase: Client, table_name: str = 'videos_data'):
    """
    Checks if there is any data present in the specified columns of the table.

    Args:
        supabase (Client): The Supabase client instance.
        table_name (str): The name of the table to query.

    Raises:
        Exception: If querying Supabase fails.
    """
    logging.info(f"Checking for data presence in table '{table_name}' for 'emotion_class' and 'video_path' columns...")

    try:
        # Check 'emotion_class' column for any non-null and non-empty entries
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

        # Check 'video_path' column for any non-null and non-empty entries
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

        # Determine overall data presence
        data_present = emotion_exists and video_exists
        logging.info(f"Overall data presence (both columns have data): {data_present}")

        # Set GitHub Actions output
        print(f"::set-output name=DATA_PRESENT::{str(data_present).lower()}")

    except Exception as e:
        logging.error(f"An error occurred while querying Supabase: {e}")
        raise

def main():
    """
    Main function to orchestrate the data presence check.
    """
    setup_logging()
    logging.info("Starting the data presence check process.")

    try:
        # Create a Supabase client
        supabase = get_supabase_client()
    except Exception as e:
        logging.critical(f"Initialization failed: {e}")
        sys.exit(1)

    try:
        check_data_presence(supabase, table_name='videos_data')  
    except Exception as e:
        logging.critical(f"Failed to check data presence: {e}")
        sys.exit(1)

    logging.info("Data presence check process completed successfully.")

if __name__ == '__main__':
    main()
