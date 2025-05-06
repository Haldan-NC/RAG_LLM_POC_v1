import pandas as pd
import time
import sys
sys.path.append(".")  
sys.path.append("..\\.")  
sys.path.append("..\\..\\.") 

from src.db.db_functions import get_cursor


def vector_embedding_cosine_similarity_search(input_text: str, chunk_size: str = "small") -> pd.DataFrame:
    """
    chunk_size: str = "small" or "large" - refers to the database table to search in.
    Searches for similar chunks based on cosine similarity.
    Returns a Pandas DataFrame with results.
    """
    if chunk_size == "small":
        table_name = "CHUNKS_SMALL"
    elif chunk_size == "large":
        table_name = "CHUNKS_LARGE"
    else:
        raise ValueError("chunk_size must be 'small' or 'large'.")

    conn,cursor = get_cursor()

    sql = f"""
    WITH input AS (
        SELECT
            SNOWFLAKE.CORTEX.EMBED_TEXT_1024('snowflake-arctic-embed-l-v2.0', %s) AS VECTOR
        )
        SELECT
            document_id,
            chunk_id,
            CHUNK_ORDER,
            PAGE_START_NUMBER,
            PAGE_END_NUMBER,
            chunk_text,
            VECTOR_COSINE_SIMILARITY({table_name}.EMBEDDING, input.VECTOR) AS COSINE_SIMILARITY
        FROM {table_name}, input
        ORDER BY COSINE_SIMILARITY DESC
        LIMIT 100
    """

    cursor.execute(sql, (input_text,))
    return_df = cursor.fetch_pandas_all()

    return return_df



def vector_embedding_cosine_similarity_between_texts(text1: str, text2: str) -> float:
    """
    Computes cosine similarity between two input texts using Snowflake Arctic Embedding.

    Args:
        text1 (str): First input text.
        text2 (str): Second input text.
        cursor: Snowflake database cursor.

    Returns:
        float: Cosine similarity between text1 and text2 (range -1 to 1).
    """

    conn,cursor = get_cursor()
    sql = f"""
    WITH embeddings AS (
    SELECT 
        SNOWFLAKE.CORTEX.EMBED_TEXT_1024('snowflake-arctic-embed-l-v2.0', %s) AS vector1,
        SNOWFLAKE.CORTEX.EMBED_TEXT_1024('snowflake-arctic-embed-l-v2.0', %s) AS vector2
    )
    SELECT VECTOR_COSINE_SIMILARITY(vector1, vector2) AS cosine_similarity
    FROM embeddings
    """

    cursor.execute(sql, (text1, text2))
    result_df = cursor.fetch_pandas_all()

    return result_df['COSINE_SIMILARITY'].iloc[0]



def extract_TOC_cortex(text: str, model : str = 'llama3.1-70b') -> str:
    """
    Extracts the table of contents from a given text using Cortex LLM.
    The function sends a prompt to the Cortex LLM and retrieves the response.

    NOTE: This function is not tested yet in this environment. It is a scaffolding code that worked in a notebook.
    """

    raise NotImplementedError("Cortex Complete in this environment is not tested yet. This is purely scaffolding code that worked in a notebook.")
    prompt = (
    f"""
    I will provide a long string of text that most likely contains a table of contents, 
    although it may also include additional body text from a document. Your task is to carefully 
    extract only the table of contents and structure it as a JSON object in the following 
    format:
    {
      "Section": "<section name>",
      "Section Number": "<section name>",
      "Page": <page number>,
      "Sub Sections" : [{
        "Section": "<section name>",
        "Section Number": "<section name>",
        "Page": <page number>,
        "Sub Sections" : []}
      ],
    }    

    Guideines:
    - All keys in the json object must be either "Section", "Section Number", "Page", "Sub Sections".
    - "Section Number" must be represented as an integer or float - E.G: 1, 2, 5.3, 1,4, etc.
    - Ignore any text that is not part of the table of contents.
    - Ensure that sub-sections are nested appropriately under their parent section.
    - Page numbers should be extracted as integers, if possible.
    - Be tolerant of inconsistencies in formatting, spacing, or punctuation (e.g. dashes, colons, ellipses).
    - Do not include duplicate or repeated sections.
    - You should only consider items which are part of the table of contents, nothing before, nothing after.
    - "Section" must consist of words
    - "Section Number" must be represented as an integer or float - E.G: 1, 2, 5.3, 1,4, etc.
    - You must include a top level key value pair called "Section":"Table of contents".

    
    Text:
    {text}
    """)


    conn, cursor = get_cursor()
    start_time = time.time()
    result = cursor.execute(f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE('{model}', $$ {prompt} $$)
    """)
    print(f"Runtime in seconds: {time.time() - start_time:.4f}")

    return cursor.fetch_pandas_all().iloc[0,0]
