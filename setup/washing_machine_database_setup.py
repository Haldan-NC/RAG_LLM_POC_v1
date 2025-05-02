
import sys
sys.path.append("..\\.")  
import yaml
import keyring
from src.db.db_functions import get_cursor


def fetch_washing_machine_pdf_files(cursor):
    pdf_files_path = "..\\data\\Washer_Manuals"
    document_rows = []

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

    cursor.execute("""
    SELECT * 
    FROM DOCUMENTS;
    """)
    documents_df = cursor.fetch_pandas_all()
    return documents_df


def create_washing_machine_schema_and_tables(cursor):
    cfg = get_config()
    database = cfg['snowflake']['database']
    schema = cfg['snowflake']['schema']
    
    try:
        cursor.execute(f" CREATE DATABASE IF NOT EXISTS {database}; ")
        cursor.execute(f" CREATE SCHEMA IF NOT EXISTS {database}.{schema}; ")
        cursor.execute(f" USE DATABASE {database}; ")
        cursor.execute(f" USE SCHEMA {schema}; ")
        return True
    except Exception as e:
        return False








if __name__ == "__main__":
    # Create Snowflake connection
    conn, cursor = get_cursor()

    # Creating the database and schema
    create_washing_machine_schema_and_tables(cursor)

    # Create a table for the documents
    fetch_washing_machine_pdf_files(cursor)


    # Create schema and tables
    # create_schema_and_tables(conn, cursor)

    # Ingest manuals
    # ingest_manuals(conn, cursor)