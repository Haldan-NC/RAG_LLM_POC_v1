import sys
sys.path.append(".")  
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
from src.utils.utils import get_connection_config, log
from src.ingestion.image_extractor import extract_images_from_pdf, generate_image_table
from src.ingestion.pdf_parser import extract_text_chunks
from src.llm_functions.open_ai_llm_functions import extract_TOC_OpenAI
from src.llm_functions.open_ai_llm_functions import call_openai_api_for_image_description


def get_cursor() -> [sf_connector, sf_connector.cursor]:
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
                                role='ACCOUNTADMIN')
    cursor = conn.cursor()

    return conn, cursor


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
        log(f"Table {table_name}, could not be written to:", level=1)
    finally:
        conn.close()
    

# def prepare_documents_df(pdf_files_path: list) -> pd.DataFrame:
#     """
#     Prepares a DataFrame of documents used for ingestion for the Documents table.
#     The reason this function is separate from the create_documents_table function is that the VGA guide is parsed seperately from the other documents.
#     Args:
#         pdf_files_path (str): Path to the directory containing PDF files.
#     """
#     document_rows = []
#     for idx, filename in enumerate(os.listdir(pdf_files_path)):
#         if filename.endswith(".pdf"):
#             file_path = os.path.join(pdf_files_path, filename)
#             log(f"Document number: {idx}  : {file_path}", level=1)
#             file_size = os.path.getsize(file_path)
            
#             document_rows.append({
#                 "DOCUMENT_NAME": filename,
#                 "FILE_PATH": file_path,
#                 "DOC_VERSION": "N/A",  # Placeholder, you can modify this printic as needed
#                 "FILE_SIZE": file_size
#             })

#     documents_df = pd.DataFrame(document_rows)
#     return documents_df

def prepare_documents_df(pdf_files_path: list) -> pd.DataFrame:
    """
    Prepares a DataFrame of documents used for ingestion for the Documents table.
    The reason this function is separate from the create_documents_table function is that the VGA guide is parsed seperately from the other documents.
    Args:
        pdf_files_path (str): Path to the directory containing PDF files.
    """
    document_rows = []
    for idx, filename in enumerate(os.listdir(pdf_files_path)):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_files_path, filename)
            log(f"Document number: {idx}  : {file_path}", level=1)
            file_size = os.path.getsize(file_path)
            

            extract_OCR_meta_data()


            extract_document_type()





            document_rows.append({
                "DOCUMENT_NAME": filename,
                "FILE_PATH": file_path,
                "DOC_VERSION": "N/A",  # Placeholder, you can modify this printic as needed
                "FILE_SIZE": file_size
            })

    documents_df = pd.DataFrame(document_rows)
    return documents_df



def extract_document_type(): 
    pass 
    

def create_OCR_meta_data(): 
    pass 












def create_documents_table(pdf_files_path: str) -> None:
    """
    Creates a Snowflake table for documents. The table is created if it does not exist.
    Args:
        pdf_files_path (str): Path to the directory containing PDF files.
    Returns:
        pd.DataFrame: DataFrame containing the documents table.
    """

    conn, cursor = get_cursor()
    document_rows = []
    cfg = get_connection_config()
    database = cfg['snowflake']['vestas']['database']
    schema = cfg['snowflake']['vestas']['schema']
    
    documents_df = prepare_documents_df(pdf_files_path)

    try: 
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
        time.sleep(1) 
        write_to_table(df = documents_df, table_name="DOCUMENTS")
    except Exception as e:
        log(f"Table DOCUMENTS, could not be created:", level=1)
    finally:
        conn.close()


def create_sections_table() -> None:
    """
    Old functions which worked with washing machine data. Requires a lot of polishing.
    """

    conn, cursor = get_cursor()
    cfg = get_connection_config()
    database = cfg['snowflake']['vestas']['database']
    schema = cfg['snowflake']['vestas']['schema']
    sections_df_list = []

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
        log("Table SECTIONS, could not be created", level=1)
    finally:
        conn.close()


def create_images_table(image_dest: str) -> None:
    """
    Creates a Snowflake table of metadata for images. The table is created if it does not exist.
    Args:
        image_dest (str): Path to the directory where the extracted images are stored.

    Returns:
        pd.DataFrame: DataFrame containing a pandas dataframe with the metadata for the images.
    """
    images_df = get_images_table()
    if type(images_df) == pd.DataFrame:
        log("Image table already exists. No need to create it again.", level=1)

    else: 
        cfg = get_connection_config()
        database = cfg['snowflake']['vestas']['database']
        schema = cfg['snowflake']['vestas']['schema']
        conn, cursor = get_cursor()
        documents_df = get_documents_table()

        all_manuals_metadata = {}
        for idx,row in tqdm(enumerate(documents_df.iterrows()), total = len(documents_df), desc = f"Extracting images from {len(documents_df)} PDFs"):
            manual_id = row[1]["DOCUMENT_ID"]
            file_path = row[1]["FILE_PATH"]
            all_manuals_metadata[manual_id] = extract_images_from_pdf(file_path, manual_id, output_dir=image_dest, verbose = 0)
            
        images_df = generate_image_table(documents_df, image_dest, all_manuals_metadata)

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
                IMAGE_X1 NUMBER,
                IMAGE_Y1 NUMBER,
                IMAGE_X2 NUMBER,
                IMAGE_Y2 NUMBER,
                DESCRIPTION STRING,
                CREATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),

                CONSTRAINT fk_document
                    FOREIGN KEY (DOCUMENT_ID)
                    REFERENCES DOCUMENTS(DOCUMENT_ID)
                    
            );
            """)

            time.sleep(1)  
            write_to_table(df = images_df, table_name = "IMAGES")
        except Exception as e:
            log(f"Table IMAGES, could not be created:", level=1)
        finally:
            conn.close()


def create_chunked_tables() -> pd.DataFrame:
    """
    Creates CHUNKS_SMALL, and CHUNKS_LARGE tables in the database.
    First it checks if the tables already exist, and if not, it creates them.

    Returns:
        2x pd.DataFrame: CHUNKS_LARGE and CHUNKS_SMALL tables.
    """
    for table_name in ["CHUNKS_LARGE", "CHUNKS_SMALL"]:
        chunks_df = get_table(table_name)
        if type(chunks_df) == pd.DataFrame:
            log(f"{table_name} table already exists. No need to create it again.", level=1)
        else:
            # This loop could potentially be improved with an enum class (JEED's suggestion)
            if table_name == "CHUNKS_LARGE":
                create_large_chunks_table()
            elif table_name == "CHUNKS_SMALL":
                create_small_chunks_table()
            else:
                log(f"Table {table_name} not found:", level=1)

    return get_large_chunk_table(), get_small_chunk_table()


def create_large_chunks_table() -> None:
    return create_windowed_chunk_table("CHUNKS_LARGE", 7000, 128)

def create_small_chunks_table() -> None:
    return create_windowed_chunk_table("CHUNKS_SMALL", 1024, 64)


def create_windowed_chunk_table(table_name: str, chunk_size: int, chunk_overlap: int) -> None:
    """Method for creating a snowflake table for chunks.
    
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
    database = cfg['snowflake']['vestas']['database']
    schema = cfg['snowflake']['vestas']['schema']
    
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
        
            log(f"Writing the {table_name} DataFrame to Snowflake", level=1)
        
        # Write the DataFrame to Snowflake
        write_to_table(df = chunks_df, table_name = table_name)
    except Exception as e:
        log(f"Table {table_name}, could not be created:", level=1)
        log(f"Exception: {e}:", level=1)

    # Add the embeddings to the chunks
    create_embeddings_on_chunks(chunks_col = "CHUNK_TEXT", table_name = table_name)


def create_embeddings_on_chunks(chunks_col: str, table_name: str) -> None:
    """
    Creates embeddings for a string column in a Snowflake table.
    It is plausible that we are interested in creating embeddings for various columns in the future.
    """
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
        log(f"Table VGA_GUIDES, could not be created:", level=1)
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
        log(f"Table VGA_GUIDE_SUBSTEPS, could not be created:", level=1)
    conn.close()


def populate_image_descriptions(sub_images_df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Populates the image descriptions in the sub_images_df DataFrame using OpenAI API.
    Args:
        sub_images_df (pd.DataFrame): DataFrame containing image information of the images found relvant to be processed with respect to the chunks of interest.
    Returns:
        pd.DataFrame: Updated DataFrame with image descriptions.
    """

    conn, cursor = get_cursor()

    # Iterate through each image and generate a description using OpenAI API
    # For each iteration, context of the image is required. It will use all small chunks of the page of the image, and the image itself.
    for idx, row in tqdm(sub_images_df.iterrows(), total=len(sub_images_df), desc="Populating image descriptions"):
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
        sub_images_df.at[idx, "DESCRIPTION"] = description_response
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

    return sub_images_df



if __name__ == "__main__":

    # The cursor works now.
    test = get_cursor()