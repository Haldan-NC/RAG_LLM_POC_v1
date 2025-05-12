
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
from src.ingestion.vga_pdf_parser import extract_vga_guide
from src.ingestion.vga_pdf_parser import * # All functions from vga_pdf_parser is inside the function process_vga_guide()


def create_vestas_schema_and_tables() -> bool:
    """
    Create the database and schema for the washing machine data in Snowflake.
    This function checks if the database and schema already exist, and if not, it creates them.
    It also sets the current database and schema to the created ones.
    Returns:
        bool: True if the database and schema were created successfully, False otherwise.
    """
    cfg = get_connection_config()
    database = cfg['snowflake']['vestas']['database']
    schema = cfg['snowflake']['vestas']['schema']
    conn, cursor = get_cursor()
    
    try:
        cursor.execute(f" CREATE DATABASE IF NOT EXISTS {database}; ")
        cursor.execute(f" CREATE SCHEMA IF NOT EXISTS {database}.{schema}; ")
        cursor.execute(f" USE DATABASE {database}; ")
        cursor.execute(f" USE SCHEMA {schema}; ")
        return True
    except Exception as e:
        return False
    finally:
        conn.close()


def create_vestas_document_table() -> pd.DataFrame:
    """
    Create the documents table for the washing machine data.

    Returns:
        pd.DataFrame: The documents DataFrame under the assumption the table was created successfully.
    """
    pdf_files_path = "data\\Vestas_RTP\\Documents\\Documents"
    document_rows = []
    conn,cursor = get_cursor()
    
    documents_df = get_documents_table()
    if type(documents_df) == pd.DataFrame:
        log("Documents table already exists. No need to create it again.", level=1)
    else:
        create_documents_table(pdf_files_path)
        documents_df = get_documents_table()

    return documents_df


def create_vestas_images_table() -> pd.DataFrame:
    """
    Create the images table for the washing machine data.
    Subprocesses include the extraction of images from the documents and the creation of the images table.
    The images are extracted from the documents and stored in a local directory specified by image_dest.
    Returns:
        pd.DataFrame: _description_
    """
    image_dest = "data\\Vestas_RTP\\Images"
    images_df = create_images_table(image_dest)

    return images_df


def create_vestas_sections_table() -> None:
    """
    This function is a placeholder for creating the sections table.
    It is currently not implemented due to the variations in Vestas documents.
    """

    #The variations of Vestas documents create poor results in the current extraction method.", level=1)
    log("The sections table will NOT be created in this version of the code!", level=1)
    return NotImplemented

    sections_df = create_sections_table()
    return sections_df


def process_vga_guide() -> None:
    """
    Process the VGA Guide document.
    This function process the VGA Guide document by creating new and appending to new and existing tables.

    # To-Do: Map images to the VGA Guide steps and substeps.
    """

    file_path = "data\\Vestas_RTP\\Documents\\VGA_guides"

    documents_df = append_vga_guide_to_documents_table()
    create_vga_guide_table()
    create_vga_guide_steps_table()
    create_vga_guide_substeps_table()

    document_id_with_vga_guide = documents_df[documents_df["FILE_PATH"].str.contains("VGA_guides")]
    for index, row in document_id_with_vga_guide.iterrows():
        document_id = row["DOCUMENT_ID"]
        file_path = row["FILE_PATH"]
        guides = extract_vga_guide(file_path = file_path, document_id = document_id)

        guides_df = create_vga_guide_dataframe(guides = guides, document_id = document_id)
        write_to_table(df = guides_df, table_name="VGA_GUIDES")
        guides_df = get_table(table_name = "VGA_GUIDES")

        steps_df = create_vga_guide_steps_dataframe(guides = guides,  guides_df = guides_df)
        write_to_table(df = steps_df, table_name="VGA_GUIDE_STEPS")

        substeps_df = create_vga_guide_substeps_dataframe(guides = guides, guides_df = guides_df)
        write_to_table(df = substeps_df, table_name="VGA_GUIDE_SUBSTEPS")


if __name__ == "__main__":

    # Creating the database and schema
    create_vestas_schema_and_tables()

    # Create a table for the documents
    documents_df = create_vestas_document_table()

    # Create chunked tables
    large_chunks_df, small_chunks_df = create_chunked_tables()

    # # Create sections table (Not implemented for Vestas / Serves as a placeholder)
    sections_df = create_vestas_sections_table()

    # # Create Images table
    images_df = create_vestas_images_table()

    # Extract VGA Guide (seperate parser from other documents)
    process_vga_guide()