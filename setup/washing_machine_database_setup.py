
import sys
sys.path.append(".")  
sys.path.append("..\\.")  
sys.path.append("..\\..\\.")  
import yaml
import keyring
import pandas as pd
from src.db.db_functions import *
from src.db.db_functions import create_chunked_tables, create_sections_table, create_images_table # Purely for the sake of readability
from src.utils.utils import get_connection_config


def create_washing_machine_schema_and_tables() -> bool:
    """
    Create the database and schema for the washing machine data in Snowflake.
    This function checks if the database and schema already exist, and if not, it creates them.
    It also sets the current database and schema to the created ones.
    Returns:
        bool: True if the database and schema were created successfully, False otherwise.
    """
    cfg = get_connection_config()
    database = cfg['snowflake']['database']
    schema = cfg['snowflake']['schema']
    conn, cursor = get_cursor()
    
    try:
        cursor.execute(f" CREATE DATABASE IF NOT EXISTS {database}; ")
        cursor.execute(f" CREATE SCHEMA IF NOT EXISTS {database}.{schema}; ")
        cursor.execute(f" USE DATABASE {database}; ")
        cursor.execute(f" USE SCHEMA {schema}; ")
        return True
    except Exception as e:
        return False


def create_washing_machine_document_table() -> pd.DataFrame:
    """
    Create the documents table for the washing machine data.

    Returns:
        pd.DataFrame: The documents DataFrame under the assumption the table was created successfully.
    """
    pdf_files_path = "data\\Washing_Machine_Data\\Documents"
    document_rows = []
    conn,cursor = get_cursor()
    
    documents_df = get_documents_table()
    if type(documents_df) == pd.DataFrame:
        log("Documents table already exists. No need to create it again.", level=1)
    else:
        documents_df = create_documents_table(pdf_files_path)

    return documents_df


def create_washing_machine_images_table() -> pd.DataFrame:
    """
    Create the images table for the washing machine data.
    Subprocesses include the extraction of images from the documents and the creation of the images table.
    The images are extracted from the documents and stored in a local directory specified by image_dest.
    Returns:
        pd.DataFrame: _description_
    """
    image_dest = "data\\Washing_Machine_Data\\Images"
    images_df = create_images_table(image_dest)

    return images_df



if __name__ == "__main__":

    # Creating the database and schema
    create_washing_machine_schema_and_tables()

    # Create a table for the documents
    documents_df = create_washing_machine_document_table()

    # Create chunked tables
    large_chunks_df, small_chunks_df = create_chunked_tables()

    # Create sections table
    sections_df = create_sections_table()

    # Create Images table
    images_df = create_washing_machine_images_table()
