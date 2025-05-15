
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
from src.utils.utils import log
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
        guides = extract_vga_guide(file_path = file_path)

        guides_df = create_vga_guide_dataframe(guides = guides, document_id = document_id)
        write_to_table(df = guides_df, table_name="VGA_GUIDES")
        # Creating embeddings for the titles of the VGA guides
        create_embeddings_on_chunks(chunks_col = "GUIDE_NAME", table_name = "VGA_GUIDES")
        guides_df = get_table(table_name = "VGA_GUIDES")

        steps_df = create_vga_guide_steps_dataframe(guides = guides,  guides_df = guides_df)
        write_to_table(df = steps_df, table_name="VGA_GUIDE_STEPS")
        # Creating embeddings for the text content of each step in a guide.
        create_embeddings_on_chunks(chunks_col = "TEXT", table_name = "VGA_GUIDE_STEPS")

        substeps_df = create_vga_guide_substeps_dataframe(guides = guides, guides_df = guides_df)
        write_to_table(df = substeps_df, table_name="VGA_GUIDE_SUBSTEPS")
        # Creating embeddings for the text content of each substep. Some of them are empty strings.
        create_embeddings_on_chunks(chunks_col = "TEXT", table_name = "VGA_GUIDE_SUBSTEPS")

        # Creates a table which links between the VGA guide steps and the DMS No. which will appear in the Documents table.
        dms_no_link_df = create_link_dataframe__step_id__dms_no(guides = guides)
        create_link_table__step_id__dms_no()
        write_to_table(df = dms_no_link_df, table_name = "LINK_STEP_DMS")


def create_wind_turbine_tables() -> None:
    """
    Creates the wind turbine tables and the links between the VGA guides and the wind turbines.
    The wind turbine tables should potentially be created in a different manner in the future.
    """
    guides_df = get_table(table_name = "VGA_GUIDES")
    turbine_df = create_wind_turbine_dataframe(guides_df = guides_df)
    create_wind_turbine_table()
    write_to_table(df = turbine_df, table_name = "WIND_TURBINES")

    turbine_link_df = create_link_dataframe__guide_id__turbine_id()
    create_link_table__guide_id__turbine_id()
    write_to_table(df = turbine_link_df, table_name = "LINK_GUIDE_TURBINE")



def create_vestas_unified_chunk_table() -> None:
    """
    Creates a unified table for all text related tables and writes it to the database.
    This function creates a unified dataframe for all text related tables.
    This dataframe will th

    Returns:
        pd.DataFrame: The unified chunk DataFrame.
    """
    log("Creating unified chunk dataframe for a table called UNION_CHUNKS...", level=1)
    
    dataframes_to_be_concatenated = []

    for table in ["CHUNKS_SMALL", "CHUNKS_LARGE"]:
        # Only getting the columns of interest from the chunked tables. Doing it this way won't break the code if new columns are added to the tables.
        chunks_df = get_table(table_name = f"{table}")[["CHUNK_ID", "DOCUMENT_ID", "CHUNK_TEXT", "CHUNK_ORDER" ,"PAGE_START_NUMBER", "PAGE_END_NUMBER"]]
        # Renaming columns to match the unified chunk table.
        chunks_df.rename(columns = {"CHUNK_TEXT": "TEXT", 
                                        "CHUNK_ID": "PREV_ID",
                                        "CHUNK_ORDER": "STEP_OR_INDEX",
                                        "PAGE_START_NUMBER": "PAGE_START",
                                        "PAGE_END_NUMBER": "PAGE_END"}, inplace = True)
        chunks_df["ORIGIN_TABLE"] = f"{table}"
        dataframes_to_be_concatenated.append(chunks_df)

    # Only VGA_GUIDES_STEPS will be included for the unified chunk table. Can be changed if required.
    vga_guide_df = get_table(table_name = "VGA_GUIDE_STEPS")[["GUIDE_STEP_ID", "DOCUMENT_ID", "STEP_LABEL", "TEXT", "STEP" ,"PAGE_START", "PAGE_END"]]
    new_text_col = [f"STEP LABEL:{x['STEP_LABEL']}\n TEXT:{x['TEXT']}" for index, x in vga_guide_df.iterrows()]
    vga_guide_df["TEXT"] = new_text_col
    vga_guide_df.drop(columns = ["STEP_LABEL"], inplace = True)
    vga_guide_df.rename(columns = {"GUIDE_STEP_ID": "PREV_ID", 
                                        "STEP": "STEP_OR_INDEX"}, inplace = True)
    vga_guide_df["ORIGIN_TABLE"] = "VGA_GUIDE_STEPS"
    dataframes_to_be_concatenated.append(vga_guide_df)


    # Iterate through the rows and create Chunk size column
    for df in dataframes_to_be_concatenated:
        df["CHUNK_SIZE"] = len(df["TEXT"].iloc[0])

    unified_chunk_df = pd.concat(dataframes_to_be_concatenated, ignore_index=True)
    
    # Create a new table in the database for the unified chunk table.
    log("Creating UNION_CHUNKS in the database...", level=1)
    conn, cursor = get_cursor()
    create_table_sql = f"""
        CREATE OR REPLACE TABLE UNION_CHUNKS (
            CHUNK_ID INT AUTOINCREMENT PRIMARY KEY,
            TEXT STRING,
            EMBEDDING VECTOR(FLOAT, 1024),
            CHUNK_SIZE INT,
            PREV_ID INT,
            ORIGIN_TABLE STRING,
            DOCUMENT_ID INT NOT NULL,
            PAGE_START INT,
            PAGE_END INT,
            STEP_OR_INDEX INT,
            CREATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),

            CONSTRAINT fk_document
                FOREIGN KEY (DOCUMENT_ID)
                REFERENCES DOCUMENTS(DOCUMENT_ID)
        );
        """
    cursor.execute(create_table_sql)
    conn.close()

    # Write the unified chunk DataFrame to the database.
    log("Writing to UNION_CHUNKS...", level=1)
    write_to_table(df = unified_chunk_df, table_name="UNION_CHUNKS")

    # Updating the embeddings for the unified chunk table.
    log("Creating embeddings for UNION_CHUNKS...", level=1)
    create_embeddings_on_chunks(chunks_col = "TEXT", table_name = "UNION_CHUNKS")


if __name__ == "__main__":

    # Creating the database and schema
    create_vestas_schema_and_tables()

    # Create a table for the documents
    documents_df = create_vestas_document_table()

    # # Create chunked tables
    # large_chunks_df, small_chunks_df = create_chunked_tables()

    # # Create Images table
    # images_df = create_vestas_images_table()

    # Extract VGA Guide (seperate parser from other documents)
    # process_vga_guide()

    # Create a unified table for all chunks
<<<<<<< HEAD
    create_vestas_unified_chunk_table()

    # Create wind turbine tables and links
    create_wind_turbine_tables()
=======
    # create_vestas_unified_chunk_table()
>>>>>>> feature/document_parser_other_documents
