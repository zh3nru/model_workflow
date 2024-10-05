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

def check_data_presence(supabase: Client, table_name: str, columns: list):
    """
    Checks if there is any data present in each specified column of the given table.
    
    Args:
        supabase (Client): The Supabase client.
        table_name (str): The name of the table to check.
        columns (list): A list of column names to verify data presence.
    """
    logging.info(f"Checking data presence in table '{table_name}' for columns: {', '.join(columns)}.")

    results = {}
    try:
        for column in columns:
            logging.info(f"Checking column '{column}' for data presence...")
            response = (
                supabase.table(table_name)
                .select(column)
                .filter(column, 'is', 'not.null')
                .filter(column, 'neq', '')  # Exclude empty strings
                .limit(1)
                .execute()
            )

            data_exists = len(response.data) > 0
            results[column] = data_exists
            logging.info(f"Data exists in column '{column}': {data_exists}")

        # Optionally, aggregate results to determine if all columns have data
        all_data_present = all(results.values())
        logging.info(f"All specified columns have data: {all_data_present}")

    except Exception as e:
        logging.error(f"An error occurred while querying Supabase: {e}")
        raise

def main():
    setup_logging()
    logging.info("Starting the data presence check process.")

    try:
        # Create a Supabase client
        supabase = get_supabase_client()
    except Exception as e:
        logging.critical(f"Initialization failed: {e}")
        sys.exit(1)

    try:
        # Specify the table and columns you want to check
        table_name = 'videos_data'
        columns_to_check = ['emotion_class', 'video_path']  # Add more columns as needed

        check_data_presence(supabase, table_name, columns_to_check)
    except Exception as e:
        logging.critical(f"Failed to check data presence: {e}")
        sys.exit(1)

    logging.info("Data presence check process completed successfully.")

if __name__ == '__main__':
    main()
