import pandas as pd
import sys
sys.path.append("..\\.")  
sys.path.append("..\\..\\.") 
sys.path.append("..\\..\\..\\.") 
from src.db.db_functions import get_cursor
from src.ingestion.llm_functions.cortex_llm_functions import vector_embedding_cosine_similarity_between_texts



def find_document_by_machine_name(machine_name: str) -> dict:
    """
    Attempts to find a document by machine name using the DOCUMENTS table.
    NOTE: THIS FUNCTION SHOULD BE REPLACED BY A BETTER FUNCTION THAT FIND MACHINE NAMES USING METADATA.
    
    First, tries simple case-insensitive substring matching against DOCUMENT_NAME.
    If no match is found, uses cosine similarity (via embeddings) to choose the best match.
    
    Args:
        cursor: Snowflake DB cursor.
        machine_name (str): The machine name to search for.
        
    Returns:
        dict: A dictionary with keys "DOCUMENT_NAME" and "DOCUMENT_ID" corresponding
              to the best matching document.
    """
    conn,cursor = get_cursor()

    # 1. Retrieve all documents
    cursor.execute("""
        SELECT DOCUMENT_NAME, DOCUMENT_ID 
        FROM DOCUMENTS;
    """)
    documents_df = cursor.fetch_pandas_all()
    
    # 2. Attempt to match using simple string matching (case insensitive)
    for _, row in documents_df.iterrows():
        doc_name = row['DOCUMENT_NAME']
        if machine_name.lower() in doc_name.lower() or doc_name.lower() in machine_name.lower():
            return {"DOCUMENT_NAME": doc_name, "DOCUMENT_ID": row["DOCUMENT_ID"]}
    
    # 3. If no string match was found, use cosine similarity via embeddings.
    best_similarity = -1.0  # cosine similarity ranges from -1 to 1.
    best_match = None
    for _, row in documents_df.iterrows():
        doc_name = row['DOCUMENT_NAME']
        # Use the cosine_similarity_between_texts function to compute similarity.
        similarity = vector_embedding_cosine_similarity_between_texts(machine_name, doc_name, cursor)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = {"DOCUMENT_NAME": doc_name, "DOCUMENT_ID": row["DOCUMENT_ID"]}
    
    return best_match


def narrow_down_relevant_chunks(task_chunk_df: pd.DataFrame, document_info: dict) -> pd.DataFrame:
    filtered_task_chunk_df = task_chunk_df[task_chunk_df['DOCUMENT_ID'] == document_info['DOCUMENT_ID']]
    filtered_task_chunk_df = filtered_task_chunk_df.sort_values(by='CHUNK_ORDER', ascending=True)
    filtered_task_chunk_df = filtered_task_chunk_df.head(10)

    return filtered_task_chunk_df


def pick_image_based_of_descriptions(image_candidates: pd.DataFrame, step_text: str) -> str:
    image_options_text = ""
    for _, image_row in image_candidates.iterrows():
        image_id = image_row["IMAGE_ID"]
        image_path = image_row["IMAGE_PATH"]
        description = image_row["DESCRIPTION"]
        image_options_text += f"- Image ID: {image_id}, Path: {image_path}, Description: {description}\n"

    instructions = f"""
    You are tasked with modifying the task in a step by step guide. You will append the most relevant image reference to the step,
    by selecting the most relevant image for the following step in a guide:
    "{step_text}"
    """

    reference_text = f"""
    ### Image Options:
    {image_options_text}
    """

    response = generate_promt_for_openai_api(instructions, input_text)
    return response.output_text