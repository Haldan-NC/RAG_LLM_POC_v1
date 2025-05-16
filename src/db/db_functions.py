import sys

sys.path.append(".")  
sys.path.append("..\\.")  
sys.path.append("..\\..\\.") 
from typing import Tuple
import os
import time
import keyring
from tqdm import tqdm
import pandas as pd
import snowflake.connector as sf_connector
from snowflake.connector.pandas_tools import write_pandas

from src.utils.utils import get_connection_config, log, SuppressStderr
from src.ingestion.image_extractor import extract_image_data_from_page
from src.ingestion.pdf_parser import extract_text_chunks
from src.llm_functions.open_ai_llm_functions import call_openai_api_for_image_description, extract_TOC_OpenAI


def get_cursor() -> Tuple[sf_connector.SnowflakeConnection, sf_connector.cursor.SnowflakeCursor]:
    """
    Returns Snowflake cursor (for RAG lookup). Called at start of RAG pipeline.

    Database and schema are hardcoded for now, but should be added to the config file.
    """
    cfg = get_connection_config()

    # These credentials need to be polished up such that they fit with the actualt crednetial manager
    account = keyring.get_password(cfg['windows_credential_manager']['snowflake']['account_identifier'], 'account_identifier')
    user = keyring.get_password(cfg['windows_credential_manager']['snowflake']['user_name'], 'user_name')
    password  = keyring.get_password(cfg['windows_credential_manager']['snowflake']['password'], 'password')
    database = cfg['snowflake']['vestas']['database']
    schema = cfg['snowflake']['vestas']['schema']

    conn = sf_connector.connect(account=account,
                                user=user, 
                                password=password, 
                                database=database,
                                schema=schema,
                                warehouse='COMPUTE_WH',
                                role='ACCOUNTADMIN',
                                disable_ocsp_checks=True,
                                insecure_mode=True
                                )
    cursor = conn.cursor()

    return conn, cursor


def log_error_table_creation(table_name: str, e: Exception) -> None:
    """Logs error when creating a table in Snowflake."""
    log(f"Table {table_name}, could not be created:", level=1)
    log(f"Exception: {e}:", level=1)


def get_table(table_name: str) -> pd.DataFrame | None:
    """SQL query execute for getting table
    
    Input:
        - table_name (str): name of table for select statment. 
    """
    conn, cursor = get_cursor()
    try:
        cursor.execute(f"""
        SELECT * 
        FROM {table_name};
        """)
        return cursor.fetch_pandas_all()
    except Exception as e:
        log(f"Table {table_name} not found:", level=1)
        return None
    finally:
        conn.close()


def get_documents_table() -> pd.DataFrame:
    return get_table("DOCUMENTS")

def get_small_chunk_table() -> pd.DataFrame:
    return get_table("CHUNKS_SMALL")

def get_large_chunk_table() -> pd.DataFrame:
        return get_table("CHUNKS_LARGE")

def get_sections_table() -> pd.DataFrame:
        return get_table("SECTIONS")
        
def get_images_table() -> pd.DataFrame:
        return get_table("IMAGES")


def remove_duplicates(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    Removes duplicates from the DataFrame based on the 'DOCUMENT_ID' and 'PAGE_START_NUMBER' columns.
    Args:
        df (pd.DataFrame): DataFrame to remove duplicates from.
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    # Compare the two tables and only keep unique rows
    # Ensure that the columns are in the same order
    snowflake_df = get_table(table_name)
    if snowflake_df is None:
        log(f"Table {table_name} not found:", level=1)
        return df

    snowflake_df = snowflake_df[df.columns]
    # Find the instances in the snowflake_df matching the df, then remove them from df
    for idx, row in df.iterrows():
        for idx2, row2 in snowflake_df.iterrows():
            if row.to_dict() == row2.to_dict():
                # log(f"Row {idx} in DataFrame already exists in Snowflake table {table_name}. Removing it from the DataFrame.", level=2)
                df.drop(idx, inplace=True)

    return df


def write_to_table(df: pd.DataFrame, table_name: str) -> None:
    """
    Appends a DataFrame to the Snowflake documents table.
    Args:
        df (pd.DataFrame): DataFrame to be appended.
    Returns:
        pd.DataFrame: Updated DataFrame containing the documents table.
    """
    conn, cursor = get_cursor()
    cfg = get_connection_config()
    database = cfg['snowflake']['vestas']['database']
    schema = cfg['snowflake']['vestas']['schema']

    # Ensure duplicated are not added 
    df = remove_duplicates(df = df, table_name = table_name)
    if len(df) == 0:
        log(f"DataFrame is empty after removing duplicates. No data to write to {table_name}.", level=1)
        return

    try: 
        success, nchunks, nrows, output = write_pandas(
            conn=conn,
            df=df,
            database=database,
            table_name=table_name,
            schema=schema,
            auto_create_table=False,
            overwrite=False
        )
        log(f"Success: {success}, Chunks: {nchunks}, Rows: {nrows}, Table Name: {table_name}", level=1)
    except Exception as e:
        log(f"Table {table_name}, could not be written to, most likely due to incorrect dataframe column names, or missing data: \nException:{e}", level=1)
    finally:
        conn.close()


def create_documents_table() -> pd.DataFrame:
    """
    Creates a Snowflake table for documents. The table is created if it does not exist.
    DMS_NO is a unique identifier for each document, which is present at all documents extracted from SAP (correct me if I'm wrong).
    The VGA guide does not have a DMS_NO on the first page.

    Returns:
        pd.DataFrame: DataFrame containing the documents table.
    """

    conn, cursor = get_cursor() 

    try: 
        cursor.execute("""
            CREATE OR REPLACE TABLE DOCUMENTS (
            DOCUMENT_ID INT AUTOINCREMENT PRIMARY KEY,
            DOCUMENT_NAME STRING,
            DMS_NO STRING, 
            VERSION STRING,
            VERSION_DATE STRING,
            EXPORTED_DATE STRING,
            DOC_TYPE STRING,
            CONFIDENTIALITY STRING,
            APPROVED STRING,
            FILE_PATH STRING NOT NULL,
            FILE_SIZE NUMBER,
            CREATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
            );
        """)
        time.sleep(1) 
    except Exception as e:
        log_error_table_creation(table_name="DOCUMENTS", e=e)
    finally:
        conn.close()


def create_wind_turbine_table() -> None:
    """
    Creates a Snowflake table for wind turbines. The table is created if it does not exist.
    """
    conn, cursor = get_cursor()
    cfg = get_connection_config()

    try: 
        cursor.execute("""
            CREATE OR REPLACE TABLE WIND_TURBINES (
            TURBINE_ID INT AUTOINCREMENT PRIMARY KEY,
            TURBINE_NAME STRING,
            SIZE STRING,
            POWER STRING,
            MK_VERSION STRING,
            CREATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
            );
        """)
        time.sleep(1) 
    except Exception as e:
        log_error_table_creation(table_name="WIND_TURBINES", e=e)

    finally:
        conn.close()


def create_link_table__guide_id__turbine_id() -> None:
    """
    Creates a Snowflake table for the link between guides and wind turbines. The table is created if it does not exist.
    """
    conn, cursor = get_cursor()
    cfg = get_connection_config()

    try: 
        cursor.execute("""
            CREATE OR REPLACE TABLE LINK_GUIDE_TURBINE (
            LINK_ID INT AUTOINCREMENT PRIMARY KEY,
            GUIDE_ID INT NOT NULL,
            TURBINE_ID INT NOT NULL,
            CREATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),

            CONSTRAINT fk_guide
                FOREIGN KEY (GUIDE_ID)
                REFERENCES VGA_GUIDES(GUIDE_ID),

            CONSTRAINT fk_turbine
                FOREIGN KEY (TURBINE_ID)
                REFERENCES WIND_TURBINES(TURBINE_ID)
            );
        """)
        time.sleep(1) 
    except Exception as e:
        log_error_table_creation(table_name="LINK_GUIDE_TURBINE", e=e)
        
    finally:
        conn.close()


def create_link_table__step_id__dms_no() -> None:
    """
    Creates a Snowflake table for the link between guides and wind turbines. The table is created if it does not exist.
    """
    conn, cursor = get_cursor()
    cfg = get_connection_config()

    try: 
        cursor.execute("""
            CREATE OR REPLACE TABLE LINK_STEP_DMS (
            LINK_ID INT AUTOINCREMENT PRIMARY KEY,
            GUIDE_ID INT NOT NULL,
            GUIDE_STEP_ID INT NOT NULL,
            DMS_NO STRING,
            DESCRIPTION STRING,
            HYPERLINK STRING,
            CREATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),

            CONSTRAINT fk_guide
                FOREIGN KEY (GUIDE_ID)
                REFERENCES VGA_GUIDES(GUIDE_ID),

            CONSTRAINT fk_guide_step
                FOREIGN KEY (GUIDE_STEP_ID)
                REFERENCES VGA_GUIDE_STEPS(GUIDE_STEP_ID)
            );
        """)
        time.sleep(1) 
    except Exception as e:
        log_error_table_creation(table_name="LINK_STEP_DMS", e=e)      

    finally:
        conn.close()


def create_sections_table() -> None:
    """
    Old functions which worked with washing machine data. Requires a lot of polishing.
    """

    conn, cursor = get_cursor()
    cfg = get_connection_config()
    sections_df_list = []

    documents_df = get_documents_table()

    # No max columns for pandas 
    pd.set_option('display.max_columns', None)

    # Get the first chunk of each document
    first_chunk_df = get_large_chunk_table().groupby("DOCUMENT_ID").first().reset_index()
    for idx, row in tqdm(enumerate(first_chunk_df.iterrows()), total = len(first_chunk_df)):
        chunk_text = row[1]["CHUNK_TEXT"]
        local_sections_df = extract_TOC_OpenAI(chunk_text)
        local_sections_df["DOCUMENT_ID"] = int(row[1]["DOCUMENT_ID"])
        sections_df_list.append(local_sections_df.copy())

    sections_df = pd.concat(sections_df_list, ignore_index=True)
    sections_df["SECTION_NUMBER"] = sections_df["SECTION_NUMBER"].astype(str)
    sections_df["PARENT_SECTION_NUMBER"] = sections_df["PARENT_SECTION_NUMBER"].astype(str)

    log(f"sections_df:\n {sections_df}", level=1)
    try: 
        cursor.execute("""
            CREATE OR REPLACE TABLE SECTIONS (
            SECTION_ID INT AUTOINCREMENT PRIMARY KEY,
            DOCUMENT_ID INT NOT NULL,
            SECTION STRING NOT NULL,
            SECTION_NUMBER STRING NOT NULL,
            PAGE INT,
            PARENT_SECTION_NUMBER STRING,
            CREATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
            CONSTRAINT fk_document
                FOREIGN KEY (DOCUMENT_ID)
                REFERENCES DOCUMENTS(DOCUMENT_ID)
        );
        """)

        time.sleep(1)  
        write_to_table(df = sections_df, table_name = "SECTIONS")
    except Exception as e:
        log_error_table_creation(table_name="SECTIONS", e=e)    

    finally:
        conn.close()
    sections_df = get_sections_table()

    return sections_df


def create_images_table() -> None:
    """
    Creates a Snowflake table of metadata for images. The table is created if it does not exist.
    Args:
        image_dest (str): Path to the directory where the extracted images are stored.

    Returns:
        pd.DataFrame: DataFrame containing a pandas dataframe with the metadata for the images.
    """
    
    cfg = get_connection_config()
    conn, cursor = get_cursor()

    try:
        cursor.execute("""
            CREATE OR REPLACE TABLE IMAGES (
            IMAGE_ID INT AUTOINCREMENT PRIMARY KEY,
            DOCUMENT_ID INT NOT NULL,
            PAGE INT,
            IMG_ORDER INT,
            IMAGE_FILE STRING,
            IMAGE_PATH STRING,
            IMAGE_SIZE NUMBER,
            IMAGE_WIDTH NUMBER,
            IMAGE_HEIGHT NUMBER,
            IMAGE_X0 NUMBER,
            IMAGE_Y0 NUMBER,
            IMAGE_X1 NUMBER,
            IMAGE_Y1 NUMBER,
            TEXT_ABOVE STRING,
            TEXT_BELOW STRING,
            TEXT_LEFT STRING,
            TEXT_RIGHT STRING,
            DESCRIPTION STRING,
            CREATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),

            CONSTRAINT fk_document
                FOREIGN KEY (DOCUMENT_ID)
                REFERENCES DOCUMENTS(DOCUMENT_ID)
                
        );
        """)
    except Exception as e:
        log_error_table_creation(table_name="IMAGES", e=e)   
    finally:
        conn.close()
    



def create_chunked_tables() -> pd.DataFrame:
    """
    Creates CHUNKS_SMALL, and CHUNKS_LARGE tables in the database.
    """
    large_chunks_df = create_large_chunks_table()
    small_chunks_df = create_small_chunks_table()

    return large_chunks_df, small_chunks_df

def create_large_chunks_table() -> pd.DataFrame:
    return create_windowed_chunk_table("CHUNKS_LARGE", 7000, 128)

def create_small_chunks_table() -> pd.DataFrame:
    return create_windowed_chunk_table("CHUNKS_SMALL", 1024, 64)


def create_windowed_chunk_table(table_name: str, chunk_size: int, chunk_overlap: int) -> pd.DataFrame:
    """Method for creating a snowflake table for chunks.
    TODO: Split this function into multiple parts, where the core components are called in vestas_database_setup.py

    Args:
        table_name: str, name of table to create
        chunk_size: int, number of characters that should be in a chunk 
        chunk_overlap: int, number of characters that should overlap between chunks
    
    Returns:
        pandas.DataFrame: panadas dataframe object representing the created table in snowflake for large chunks
    """
    
    conn, cursor = get_cursor()
    documents_df = get_documents_table()
    # Get Config    
    cfg = get_connection_config()
    
    create_table_sql = f"""
    CREATE OR REPLACE TABLE {table_name} (
        CHUNK_ID INT AUTOINCREMENT PRIMARY KEY,
        DOCUMENT_ID INT NOT NULL,
        PAGE_START_NUMBER INT,
        PAGE_END_NUMBER INT,
        CHUNK_ORDER INT,
        CHUNK_TEXT STRING NOT NULL,
        EMBEDDING VECTOR(FLOAT, 1024),
        CREATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
        CONSTRAINT fk_document
            FOREIGN KEY (DOCUMENT_ID)
            REFERENCES DOCUMENTS(DOCUMENT_ID)
    );
    """
    cursor.execute(create_table_sql) # Creating the table without data
    try:
    
        chunks_df = pd.DataFrame()
        for row in tqdm(documents_df.iterrows(), total = len(documents_df), desc = f"Creating chunks for {table_name}"):
            manual_id = row[1]["DOCUMENT_ID"]
            file_path = row[1]["FILE_PATH"]
            tmp_chunked_df = extract_text_chunks(file_path = file_path,
                                manual_id = manual_id,
                                chunk_size = chunk_size,#1024,
                                chunk_overlap = chunk_overlap)  # Show first 5 chunks
            chunks_df = pd.concat([chunks_df, tmp_chunked_df], ignore_index=True)
        
        # Write the DataFrame to Snowflake
        log(f"Writing the {table_name} DataFrame to Snowflake", level=1)
        write_to_table(df = chunks_df, table_name = table_name)
    except Exception as e:
        log_error_table_creation(table_name=""+table_name, e=e)
    finally:
        conn.close()

        # Add the embeddings to the chunks
        create_embeddings_on_chunks(chunks_col = "CHUNK_TEXT", table_name = table_name)
        chunks_df = get_table(table_name)

    return chunks_df


def create_embeddings_on_chunks(chunks_col: str, table_name: str) -> bool:
    conn, cursor = get_cursor()
    try:
        # Update the embeddings for the chunks in the CHUNKS_LARGE table
        cursor.execute(f"""
            UPDATE {table_name}
            SET EMBEDDING = SNOWFLAKE.CORTEX.EMBED_TEXT_1024(
                'snowflake-arctic-embed-l-v2.0',
                {chunks_col}
            )
            WHERE EMBEDDING IS NULL;
        """)
        conn.close()
    except Exception as e:
        log(f"Table {table_name}, could not be updated with EMBEDDING:", level=1)
        conn.close()


def create_vga_guide_table() -> None:
    """
    A function to create the VGA_GUIDES table in the database.
    The table is created with the intention of structuring the information of the VGA guides.
    It might possibly be redundant in the great scheme of things, but it allows for flexibility in the future.

    This table stores the metadata for each guide found in the VGA guide document, 
        such as the guide name, guide number, page number, and wind turbine models for each guide. 
    """
    try:
        conn, cursor = get_cursor()
        create_table_sql = f"""
            CREATE OR REPLACE TABLE VGA_GUIDES (
                GUIDE_ID INT AUTOINCREMENT PRIMARY KEY,
                DOCUMENT_ID INT NOT NULL,
                GUIDE_NUMBER INT NOT NULL,
                PAGE_NUMBER INT,
                GUIDE_NAME STRING NOT NULL,
                STEPS INT,
                TURBINE_MODELS STRING,
                EMBEDDING VECTOR(FLOAT, 1024),
                CREATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
                CONSTRAINT fk_document
                    FOREIGN KEY (DOCUMENT_ID)
                    REFERENCES DOCUMENTS(DOCUMENT_ID)
            );
            """
        cursor.execute(create_table_sql)
    except Exception as e:
        log_error_table_creation(table_name="VGA_GUIDES", e=e)
        
    conn.close()


def create_vga_guide_steps_table() -> None:
    """
    A function to create the VGA_GUIDE_STEPS table in the database.
    The table is created with the intention of structuring the information of the VGA guides.
    It might possibly be redundant in the great scheme of things, but it allows for flexibility in the future.

    This table stores the concatenated text data for each step in each guide, as each steps can have multiple substeps.
        Concatenating the data for each step might allow for better performance when the text is embedded for vector search.
    
    """
    try:
        conn, cursor = get_cursor()
        create_table_sql = f"""
            CREATE OR REPLACE TABLE VGA_GUIDE_STEPS (
                GUIDE_STEP_ID INT AUTOINCREMENT PRIMARY KEY,
                DOCUMENT_ID INT NOT NULL,
                GUIDE_ID INT NOT NULL,
                GUIDE_NUMBER INT NOT NULL,
                PAGE_START INT,
                PAGE_END INT,
                STEP INT,
                STEP_LABEL STRING,
                TEXT STRING,
                EMBEDDING VECTOR(FLOAT, 1024),
                CREATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),

                CONSTRAINT fk_guide
                    FOREIGN KEY (GUIDE_ID)
                    REFERENCES VGA_GUIDES(GUIDE_ID),

                CONSTRAINT fk_document
                    FOREIGN KEY (DOCUMENT_ID)
                    REFERENCES DOCUMENTS(DOCUMENT_ID)
            );
            """
        cursor.execute(create_table_sql)
    except Exception as e:
        log(f"Table VGA_GUIDE_STEPS, could not be created:", level=1)
        log_error_table_creation(table_name="VGA_GUIDE_STEPS", e=e)
    conn.close()


def create_vga_guide_substeps_table() -> None:
    """
    A function to create the VGA_GUIDE_STEPS table in the database.
    The table is created with the intention of structuring the information of the VGA guides.
    It might possibly be redundant in the great scheme of things, but it allows for flexibility in the future.

    This table stores the text data for each sub step in each step, in each guide.
        Having a seperate table for the substeps can help us with mapping the images to the relevant substeps.
    """
    try:
        conn, cursor = get_cursor()
        create_table_sql = f"""
            CREATE OR REPLACE TABLE VGA_GUIDE_SUBSTEPS (
                GUIDE_SUBSTEP_ID INT AUTOINCREMENT PRIMARY KEY,
                DOCUMENT_ID INT NOT NULL,
                GUIDE_ID INT NOT NULL,
                GUIDE_NUMBER INT NOT NULL,
                PAGE_NUMBER INT,
                STEP INT,
                STEP_LABEL STRING,
                TEXT STRING,
                EMBEDDING VECTOR(FLOAT, 1024),
                CREATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),

                CONSTRAINT fk_guide
                    FOREIGN KEY (GUIDE_ID)
                    REFERENCES VGA_GUIDES(GUIDE_ID),

                CONSTRAINT fk_document
                    FOREIGN KEY (DOCUMENT_ID)
                    REFERENCES DOCUMENTS(DOCUMENT_ID)
            );
            """
        cursor.execute(create_table_sql)
    except Exception as e:
        log_error_table_creation(table_name="VGA_GUIDE_SUBSTEPS", e=e)
    conn.close()

    

def populate_image_descriptions(images_df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Populates the image descriptions in the images_df DataFrame using OpenAI API.
    Args:
        images_df (pd.DataFrame): DataFrame containing image information.
    Returns:
        pd.DataFrame: Updated DataFrame with image descriptions.
    """

    conn, cursor = get_cursor()

    # Iterate through each image and generate a description using OpenAI API
    # For each iteration, context of the image is required. It will use all small chunks of the page of the image, and the image itself.
    for idx, row in tqdm(images_df.iterrows(), total=len(images_df), desc="Populating image descriptions"):
        if len(row["DESCRIPTION"]) > 0:
            log(f"Image ID {row['IMAGE_ID']} already has a description. Skipping...", level=1)
            continue # Skip if description already exists
        log(f"Generating description for image ID {row['IMAGE_ID']}...", level=1)

        file_location = row["IMAGE_PATH"]
        page_number = row["PAGE"]
        document_id = row["DOCUMENT_ID"]
        image_id = row["IMAGE_ID"]

        sql = f"""
        SELECT * 
        FROM CHUNKS_SMALL 
        WHERE PAGE_START_NUMBER = %s AND DOCUMENT_ID = %s
        """

        # Important: pass input_text as a parameter, NOT interpolated directly
        cursor.execute(sql, (page_number,document_id,))
        local_small_chunks = cursor.fetch_pandas_all()

        # Create a context string from the relevant chunks
        context_string = "\n".join(local_small_chunks["CHUNK_TEXT"].tolist())
        prompt = f"""
            This image was extracted from the same page as the context string which is concatenated at the end of this string. 
            Please describe the image, including any relevant information that can be inferred from the context.
            The description should be consise, information dense, and mostly relevant to the image rather than the parsed context.

            CONTEXT:
            {context_string}
            """

        # Call OpenAI API to generate a description for the image
        description_response = call_openai_api_for_image_description(file_location, prompt)

        # Store the generated description in the DataFrame
        images_df.at[idx, "DESCRIPTION"] = description_response
        log(f"Updated IMAGE table for image ID:{image_id} with new description", level=1)

        # Update the database with the new description
        update_sql = f"""
        UPDATE IMAGES
        SET DESCRIPTION = %s
        WHERE IMAGE_ID = %s
        """
        cursor.execute(update_sql, (description_response, image_id))
        cursor.connection.commit()
        conn.close()
    return images_df

def drop_database() -> None:
    """
    Drops the database and all its tables.
    """
    conn, cursor = get_cursor()
    cfg = get_connection_config()
    database = cfg['snowflake']['vestas']['database']

    try:
        cursor.execute(f"DROP DATABASE {database};")
        log(f"Database {database} dropped successfully.", level=1)
    except Exception as e:
        log(f"Error dropping database {database}: {e}", level=1)
    finally:
        conn.close()

def drop_database() -> None:
    """
    Drops the database and all its tables.
    """
    conn, cursor = get_cursor()
    cfg = get_connection_config()
    database = cfg['snowflake']['vestas']['database']

    try:
        cursor.execute(f"DROP DATABASE {database};")
        log(f"Database {database} dropped successfully.", level=1)
    except Exception as e:
        log(f"Error dropping database {database}: {e}", level=1)
    finally:
        conn.close()


if __name__ == "__main__":

    # The cursor works now.
    test = get_cursor()