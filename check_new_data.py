import os
from supabase import create_client, Client
import logging
import sys

def setup_logging():
    """
    Sets up the logging configuration.
    """
    logging.basicConfig(
        level=logging.INFO,  # You can change this to DEBUG for more detailed logs
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_supabase_client() -> Client:
    """
    Retrieves Supabase credentials from environment variables and creates a Supabase client.

    Returns:
        Client: An instance of the Supabase client.

    Raises:
        EnvironmentError: If the Supabase credentials are not found.
        Exception: If the client creation fails.
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

def check_new_data(supabase: Client, table_name: str = 'videos_data'):
    """
    Checks if there is any data in the specified columns ('emotion_class' and 'video_path').

    Args:
        supabase (Client): Supabase client instance.
        table_name (str): Name of the table to query.
    """
    logging.info(f"Checking for data presence in table '{table_name}' for 'emotion_class' and 'video_path' columns...")

    try:
        # Query Supabase for records where 'emotion_class' and 'video_path' are not null and not empty
        response = (
            supabase.table(table_name)
            .select('id')
            .filter('emotion_class', 'neq', None)  # Check if 'emotion_class' is NOT NULL
            .filter('video_path', 'neq', None)     # Check if 'video_path' is NOT NULL
            .filter('emotion_class', 'neq', '')    # Ensure 'emotion_class' is not an empty string
            .filter('video_path', 'neq', '')       # Ensure 'video_path' is not an empty string
            .limit(1)
            .execute()
        )

        data_exists = len(response.data) > 0
        logging.info(f"Data exists: {data_exists}")

        # Print the result so that the workflow can capture it
        print(f"::set-output name=NEW_DATA::{str(data_exists).lower()}")

    except Exception as e:
        logging.error(f"An error occurred while querying Supabase: {e}")
        raise

def main():
    """
    The main function orchestrates the setup and execution of data checking.
    """
    setup_logging()
    logging.info("Starting the new data check process.")

    try:
        # Create a Supabase client
        supabase = get_supabase_client()
    except Exception as e:
        logging.critical(f"Initialization failed: {e}")
        sys.exit(1)

    # Call the check_new_data function with the correct table name
    try:
        check_new_data(supabase, table_name='videos_data')  # Check data presence
    except Exception as e:
        logging.critical(f"Failed to check new data: {e}")
        sys.exit(1)

    logging.info("New data check process completed successfully.")

if __name__ == '__main__':
    main()
