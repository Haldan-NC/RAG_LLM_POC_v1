
import sys
sys.path.append("..\\.")  
import yaml
import keyring
import pandas as pd
from src.db.db_functions import *
from src.db.db_functions import create_chunked_tables, create_sections_table, create_images_table # Purely for the sake of readability
from src.utils.utils import get_config


def create_washing_machine_schema_and_tables():
    cfg = get_config()
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


def create_washing_machine_document_table():
    pdf_files_path = "..\\data\\Washing_Machine_Data\\Documents"
    document_rows = []
    conn,cursor = get_cursor()
    
    documents_df = get_documents_table()
    if type(documents_df) == pd.DataFrame:
        print("Documents table already exists. No need to create it again.")
    else:
        documents_df = create_documents_table(pdf_files_path)

    return documents_df


def create_washing_machine_images_table():
    pdf_source = "..\\data\\Washing_Machine_Data\\Documents"
    image_dest = "..\\data\\Washing_Machine_Data\\images"
    images_df = create_images_table(image_source, image_dest)

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
    
