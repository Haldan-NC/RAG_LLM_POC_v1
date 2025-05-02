import re
import json
import time
from langchain_community.document_loaders import PDFPlumberLoader
import pandas as pd
from tqdm import tqdm
from src.ingestion.text_extractor import ocr_extract
from snowflake.connector.pandas_tools import write_pandas # (https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-api#write_pandas)
from src.utils.utils import get_config


def extract_text_chunks(file_path: str, manual_id: int, chunk_size: int = 512, chunk_overlap: int = 128) -> pd.DataFrame:
    """
    Extracts text chunks from a PDF file, tracking the page numbers and creating a DataFrame.
    Args:
        file_path (str): Path to the PDF file.
        manual_id (int): Manual ID for the document.
        chunk_size (int): Size of each text chunk.
        chunk_overlap (int): Overlap between chunks.
    """
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # Step 1: Combine all text across pages with page tracking
    all_text = ""
    page_map = []  # (char_index, page_number)

    for doc_page in docs:
        text = doc_page.page_content.strip().replace('\n', ' ')
        start_idx = len(all_text)
        all_text += text + " "  # Add space to separate pages
        end_idx = len(all_text)
        page_map.append((start_idx, end_idx, doc_page.metadata['page']))

    # Step 2: Create chunks with overlap, spanning across pages
    chunks = []
    chunk_order = []
    page_start_list = []
    page_end_list = []

    idx = 0
    chunk_idx = 0

    while idx < len(all_text):
        chunk = all_text[idx:idx + chunk_size]

        # Determine pages involved in this chunk
        chunk_start = idx
        chunk_end = idx + len(chunk)

        pages_in_chunk = [
            page_num
            for start, end, page_num in page_map
            if not (end <= chunk_start or start >= chunk_end)  # overlap condition
        ]

        page_start = min(pages_in_chunk) if pages_in_chunk else None
        page_end = max(pages_in_chunk) if pages_in_chunk else None

        chunks.append(chunk)
        page_start_list.append(page_start)
        page_end_list.append(page_end)
        chunk_order.append(chunk_idx)

        chunk_idx += 1
        idx += chunk_size - chunk_overlap

    # Step 3: Create DataFrame
    rows = [{
        'DOCUMENT_ID': manual_id,
        'PAGE_START_NUMBER': start,
        'PAGE_END_NUMBER': end,
        'CHUNK_TEXT': chunk,
        'CHUNK_ORDER': order
    } for chunk, start, end, order in zip(chunks, page_start_list, page_end_list, chunk_order)]

    df = pd.DataFrame(rows, columns=["DOCUMENT_ID", "PAGE_START_NUMBER", "PAGE_END_NUMBER", "CHUNK_TEXT", "CHUNK_ORDER"])
    return df

def get_chunk_table(cursor, table_name):
    cursor.execute(f"""
    SELECT * 
    FROM {table_name};
    """)

# Table names are either CHUNKS_LARGE or CHUNKS_SMALL
# Chunk size is either 7000 or 1024
# Chunk overlapp is 128 or 64

def create_cunk_table(conn, documents_df, table_type) -> pd:
    """Method for creating a snowflake table for chunks.
    
    Input:
        - conn: A snowflake Connection object holding the connection and session information to keep the database connection active.
        - documents_df: Pandas.DataFrame, documents table 
    
    Returns:
        pandas.DataFrame: panadas dataframe object representing the created table in snowflake for large chunks
    """
    if table_type == "LARGE":
        table_name = "CHUNKS_LARGE"
        chunk_size = 7000
        chunk_overlap = 128
    elif table_type == "SMALL":
        table_name = "CHUNKS_SMALL"
        chunk_size = 1024
        chunk_overlap = 64
    else: 
        raise ValueError(f"table type {table_type} is not allowed")
    
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
    cursor = conn.cursor()
    cursor.execute(create_table_sql)
    
    large_chunks_df = pd.DataFrame()
    for row in tqdm(documents_df.iterrows(), total = len(documents_df)):
        manual_id = row[1]["DOCUMENT_ID"]
        file_path = row[1]["FILE_PATH"]
        tmp_chunked_df = extract_text_chunks(file_path = file_path, 
                            manual_id = manual_id,
                            chunk_size = chunk_size,#1024,
                            chunk_overlap = chunk_overlap)  # Show first 5 chunks
        large_chunks_df = pd.concat([large_chunks_df, tmp_chunked_df], ignore_index=True)
    
        print("Writing the large chunks DataFrame to Snowflake")
        
    # Get Config    
    cfg = get_config()
    database = cfg['snowflake']['database']
    schema = cfg['snowflake']['schema']
    
    
    # Write the DataFrame to Snowflake
    success, nchunks, nrows, output = write_pandas(
        conn=conn,  # Convert conn, database objects to a object? 
        df=large_chunks_df,
        database =database,
        table_name=table_name,
        schema=schema,
        auto_create_table=False,
        overwrite=False
    )
    
    
    print(f"Success: {success}, Chunks: {nchunks}, Rows: {nrows}")
    time.sleep(2)

    # Update the embeddings for the chunks in the CHUNKS_LARGE table
    cursor.execute("""
        UPDATE CHUNKS_LARGE
        SET EMBEDDING = SNOWFLAKE.CORTEX.EMBED_TEXT_1024(
            'snowflake-arctic-embed-l-v2.0',
            CHUNK_TEXT
        )
        WHERE EMBEDDING IS NULL;
    """)

    time.sleep(2)
    large_chunks_df = get_chunk_table(cursor, table_name)
    
    return large_chunks_df


if __name__ == "__main__":
    # Example usage
    text = "1. Introduction 1\n2. Methodology 2\n3. Results 3\n4. Conclusion 4"
    toc = extract_TOC_OpenAI(text)
    print(toc)  # This will print the extracted table of contents in JSON format.