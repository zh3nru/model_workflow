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

    # Retrieves Supabase credentials and creates Supabase client

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

def check_new_data(supabase: Client, table_name: str = 'videos_data'):
    
    # Check data.
    logging.info(f"Checking for data presence in table '{table_name}' for 'emotion_class' and 'video_path' columns...")

    try:
        # Query Supabase for records
        response = (
            supabase.table(table_name)
            .select('id')
            .filter('emotion_class', 'neq', None)  
            .filter('video_path', 'neq', None)     
            .filter('emotion_class', 'neq', '')    
            .filter('video_path', 'neq', '')       
            .limit(1)
            .execute()
        )

        data_exists = len(response.data) > 0
        logging.info(f"Data exists: {data_exists}")

        print(f"::set-output name=NEW_DATA::{str(data_exists).lower()}")

    except Exception as e:
        logging.error(f"An error occurred while querying Supabase: {e}")
        raise

def main():
   
    setup_logging()
    logging.info("Starting the new data check process.")

    try:
        # Create a Supabase client
        supabase = get_supabase_client()
    except Exception as e:
        logging.critical(f"Initialization failed: {e}")
        sys.exit(1)

    try:
        check_new_data(supabase, table_name='videos_data')  
    except Exception as e:
        logging.critical(f"Failed to check new data: {e}")
        sys.exit(1)

    logging.info("New data check process completed successfully.")

if __name__ == '__main__':
    main()
