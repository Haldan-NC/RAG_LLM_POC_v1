import sys
sys.path.append("..\\.")  
sys.path.append("..\\..\\.") 
import os
import time
import yaml
import keyring
from typing import Optional
from tqdm import tqdm
import pandas as pd
import snowflake.connector as sf_connector
from snowflake.connector.pandas_tools import write_pandas
from src.utils.utils import get_config
from src.ingestion.llm_functions.open_ai_llm_functions import extract_TOC_OpenAI
from src.ingestion.image_extractor import extract_images_from_pdf, generate_image_table
from src.ingestion.pdf_parser import extract_text_chunks


def get_connection(account: str, user: str, password: str, database: str, schema: str):
    """
    Returns a Snowflake connection and cursor. Used by setup and RAG client.
    """
    conn = sf_connector.connect(account=account,
                                user=user, 
                                password=password, 
                                database=database,
                                schema=schema,
                                warehouse='COMPUTE_WH',
                                role='ACCOUNTADMIN')
    cursor = conn.cursor()
    return conn, cursor


def get_cursor():
    """
    Returns Snowflake cursor (for RAG lookup). Called at start of RAG pipeline.

    Database and schema are hardcoded for now, but should be added to the config file.
    """
    cfg = get_config()

    # These credentials need to be polished up such that they fit with the actualt crednetial manager
    acct = keyring.get_password(cfg['windows_credential_manager']['snowflake']['account_identifier'], 'account_identifier')
    user = keyring.get_password(cfg['windows_credential_manager']['snowflake']['user_name'], 'user_name')
    pwd  = keyring.get_password(cfg['windows_credential_manager']['snowflake']['password'], 'password')
    database = cfg['snowflake']['database']
    schema = cfg['snowflake']['schema']

    conn, cursor = get_connection(acct, user, pwd, database, schema)
    return conn, cursor


def get_table(table_name) -> Optional[pd]:
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
        print(f"Table {table_name} not found:")
        return None


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


def create_documents_table(pdf_files_path: str) -> pd.DataFrame:
    """
    Creates a Snowflake table for documents. The table is created if it does not exist.
    Args:
        pdf_files_path (str): Path to the directory containing PDF files.
    Returns:
        pd.DataFrame: DataFrame containing the documents table.
    """
    document_rows = []
    conn, cursor = get_cursor()
    cfg = get_config()
    database = cfg['snowflake']['database']
    schema = cfg['snowflake']['schema']
    documents_df = get_documents_table()

    if type(documents_df) == pd.DataFrame:
        print("Documents table already exists. No need to create it again.")
    else:

        for idx, filename in enumerate(os.listdir(pdf_files_path)):
            if filename.endswith(".pdf"):
                file_path = os.path.join(pdf_files_path, filename)
                print(f"Document number: {idx}  : {file_path}")
                file_size = os.path.getsize(file_path)
                
                document_rows.append({
                    "DOCUMENT_NAME": filename,
                    "FILE_PATH": file_path,
                    "DOC_VERSION": "N/A",  # Placeholder, you can modify this logic as needed
                    "FILE_SIZE": file_size
                })

        cursor.execute("""
            CREATE OR REPLACE TABLE DOCUMENTS (
            DOCUMENT_ID INT AUTOINCREMENT PRIMARY KEY,
            DOCUMENT_NAME STRING,
            DOC_VERSION STRING,
            FILE_PATH STRING NOT NULL,
            FILE_SIZE NUMBER,
            CREATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
            );
        """)
        time.sleep(2)  # Sleep for 2 seconds to ensure the table is ready in snowflake. We need to query the table to get the DOCUMENT_ID

        documents_df = pd.DataFrame(document_rows)
        success, nchunks, nrows, output = write_pandas(
            conn=conn,
            df=documents_df,
            database=database,
            table_name="DOCUMENTS",
            schema=schema,
            auto_create_table=False,
            overwrite=False
        )
        print(f"Success: {success}, Chunks: {nchunks}, Rows: {nrows}")
        time.sleep(2)  # Sleep for 3 seconds to ensure the table is ready in snowflake. We need to query the table to get the DOCUMENT_ID

        documents_df = get_documents_table()
    return documents_df


def create_sections_table():
    sections_df = get_sections_table()
    if type(sections_df) == pd.DataFrame:
        print("Sections table already exists. No need to create it again.")
    else:
        conn, cursor = get_cursor()
        cfg = get_config()
        database = cfg['snowflake']['database']
        schema = cfg['snowflake']['schema']
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

        print("sections_df:\n", sections_df)
        print("")

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

        time.sleep(2)  
        success, nchunks, nrows, output = write_pandas(
            conn=conn,
            df=sections_df,
            database =database,
            table_name="SECTIONS",
            schema=schema,
            auto_create_table=False,
            overwrite=False
        )
        print(f"Success: {success}, Chunks: {nchunks}, Rows: {nrows}")

        time.sleep(3) # Sleep for 3 seconds to ensure the table is ready in snowflake. We need to query the table to get the SECTION_ID
        sections_df = get_sections_table()

    return sections_df


def create_images_table(image_dest: str) -> pd.DataFrame:

    # Get db connection
    cfg = get_config()
    database = cfg['snowflake']['database']
    schema = cfg['snowflake']['schema']
    conn, cursor = get_cursor()

    images_df = get_images_table()
    if type(images_df) == pd.DataFrame:
        print("Image table already exists. No need to create it again.")

    else: 
        documents_df = get_documents_table()
        sections_df = get_sections_table()

        all_manuals_metadata = {}
        for idx,row in tqdm(enumerate(documents_df.iterrows()), total = len(documents_df)):
            manual_id = row[1]["DOCUMENT_ID"]
            file_path = row[1]["FILE_PATH"]
            all_manuals_metadata[manual_id] = extract_images_from_pdf(file_path, manual_id, output_dir=image_dest, verbose = 0)
            
        images_df = generate_image_table(documents_df, sections_df, image_dest, all_manuals_metadata)

        cursor.execute("""
            CREATE OR REPLACE TABLE IMAGES (
            IMAGE_ID INT AUTOINCREMENT PRIMARY KEY,
            SECTION_ID INT NOT NULL,
            DOCUMENT_ID INT NOT NULL,
            SECTION_NUMBER STRING NOT NULL,
            PAGE INT,
            IMG_ORDER INT,
            IMAGE_FILE STRING,
            IMAGE_PATH STRING,
            IMAGE_SIZE NUMBER,
            IMAGE_WIDTH NUMBER,
            IMAGE_HEIGHT NUMBER,
            IMAGE_X1 NUMBER,
            IMAGE_Y1 NUMBER,
            IMAGE_X2 NUMBER,
            IMAGE_Y2 NUMBER,
            DESCRIPTION STRING,
            CREATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),

            CONSTRAINT fk_document
                FOREIGN KEY (DOCUMENT_ID)
                REFERENCES DOCUMENTS(DOCUMENT_ID),
                
            CONSTRAINT fk_section
                    FOREIGN KEY (SECTION_ID)
                    REFERENCES SECTIONS(SECTION_ID)
        );
        """)

        time.sleep(2)  
        success, nchunks, nrows, output = write_pandas(
            conn=conn,
            df=images_df,
            database =database,
            table_name="IMAGES",
            schema=schema,
            auto_create_table=False,
            overwrite=False
        )

        time.sleep(2) # Sleep for 2 seconds to ensure the table is ready in snowflake. We need to query the table to get the SECTION_ID
        images_df = get_images_table()

    return images_df



def create_chunked_tables() -> pd.DataFrame:
    """
    Creates CHUNKS_SMALL, and CHUNKS_LARGE tables in the database.
    """
    large_chunks_df = create_large_chunks_table()
    
    small_chunks_df = create_small_chunks_table()

    return large_chunks_df, small_chunks_df


def create_large_chunks_table() -> pd:
    return create_chunk_table("CHUNKS_LARGE", 7000, 128)

def create_small_chunks_table() -> pd:
    return create_chunk_table("CHUNKS_SMALL", 1024, 64)


def create_chunk_table(table_name: str, chunk_size: int, chunk_overlap: int) -> pd:
    """Method for creating a snowflake table for chunks.
    
    Args:
        table_name: str, name of table to create
        chunk_size: int, number of characters that should be in a chunk 
        chunk_overlap: int, number of characters that should overlap between chunks
    
    Returns:
        pandas.DataFrame: panadas dataframe object representing the created table in snowflake for large chunks
    """
    chunks_df = get_table(table_name)
    if type(chunks_df) == pd.DataFrame:
        print(f"{table_name} table already exists. No need to create it again.")
    else:

        conn, cursor = get_cursor()
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
        cursor.execute(create_table_sql)
        
        documents_df = get_documents_table()
        chunks_df = pd.DataFrame()
        for row in tqdm(documents_df.iterrows(), total = len(documents_df)):
            manual_id = row[1]["DOCUMENT_ID"]
            file_path = row[1]["FILE_PATH"]
            tmp_chunked_df = extract_text_chunks(file_path = file_path,
                                manual_id = manual_id,
                                chunk_size = chunk_size,#1024,
                                chunk_overlap = chunk_overlap)  # Show first 5 chunks
            chunks_df = pd.concat([chunks_df, tmp_chunked_df], ignore_index=True)
        
            print("Writing the large chunks DataFrame to Snowflake")
            
        # Get Config    
        cfg = get_config()
        database = cfg['snowflake']['database']
        schema = cfg['snowflake']['schema']
        
        
        # Write the DataFrame to Snowflake
        success, nchunks, nrows, output = write_pandas(
            conn=conn,  # Convert conn, database objects to a object? 
            df=chunks_df,
            database =database,
            table_name=table_name,
            schema=schema,
            auto_create_table=False,
            overwrite=False
        )
        
        
        print(f"Success: {success}, Chunks: {nchunks}, Rows: {nrows}")
        time.sleep(2)

        # Update the embeddings for the chunks in the CHUNKS_LARGE table
        cursor.execute(f"""
            UPDATE {table_name}
            SET EMBEDDING = SNOWFLAKE.CORTEX.EMBED_TEXT_1024(
                'snowflake-arctic-embed-l-v2.0',
                CHUNK_TEXT
            )
            WHERE EMBEDDING IS NULL;
        """)

        time.sleep(2)
        chunks_df = get_table(table_name)
    
    return chunks_df



if __name__ == "__main__":

    # The cursor works now.
    test = get_cursor()