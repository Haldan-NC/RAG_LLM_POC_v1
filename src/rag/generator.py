import sys
import pandas as pd
sys.path.append("..\\.")  
sys.path.append("..\\..\\.") 
sys.path.append("..\\..\\..\\.") 

from src.db.db_functions import get_cursor
from src.db.db_functions import populate_image_descriptions



def add_image_references_to_guide(guide_text: str, filtered_task_chunk_df: pd.DataFrame) -> str:
    """
    Inserts image references into a step-by-step guide based on LLM-evaluated image descriptions.

    Args:
        guide_text (str): Step-by-step markdown text.
        filtered_task_chunk_df (pd.DataFrame): Chunks used to build the guide.

    Returns:
        str: Guide text with images inserted into appropriate steps.
    """

    # Placeholder until the code has been tested and cleaned up.
    conn, cursor = get_cursor()

    # Populate all images with descriptions on the pages where the relevant chunks are located.
    relevant_pages = filtered_task_chunk_df["PAGE_START_NUMBER"].unique()
    document_id = int(filtered_task_chunk_df["DOCUMENT_ID"].iloc[0]) # Assumes that only 1 document is relevant for the task.

    sql = f"""
        SELECT * 
        FROM IMAGES 
        WHERE PAGE IN ({','.join(map(str, relevant_pages))})
        AND DOCUMENT_ID = %s
    """
    cursor.execute(sql, (document_id,))
    images_df = cursor.fetch_pandas_all()

    print("Populating image descriptions if they don't exist...")
    images_df = populate_image_descriptions(images_df)

    image_descriptors = create_image_string_descriptors(images_df)
    user_query = f"{guide_text} \n \n {image_descriptors}"
    print("Calling OpenAI API to add image references to the guide...")

    response = generate_promt_for_openai_api(
        instructions="""
        You are are tasked to modify the step by step guide below, and include the most relevant images.
        You should only include images IFF they are relevant to the step.
        The way you will do this is by adding the IMAGE_ID and PATH to the step.
        The input_text will include a long description of each candidate image, and the step by step guide.
        """, 
        input_text = user_query
        )

    return response.output_text



def create_step_by_step_prompt(relevant_chunks_df: pd.DataFrame, user_task: str) -> str:
    """
    Builds a prompt asking the LLM to create a step-by-step guide based on relevant chunks.
    
    Args:
        relevant_chunks_df (pd.DataFrame): DataFrame of retrieved relevant chunks.
        user_task (str): The original user query (e.g., "How do I clean the filter?")
    
    Returns:
        str: Prompt ready for LLM completion
    """

    reference_text = ""
    for i, row in relevant_chunks_df.iterrows():
        page_info = f"(page {row['PAGE_START_NUMBER']})" if 'PAGE_START_NUMBER' in row else ""
        reference_text += f"- [Relevance: {row['COSINE_SIMILARITY']}]: {row['CHUNK_TEXT']}  {page_info}\n\n"
        # Section info could also be included in the prompt if needed.

    instructions = f"""
    You are tasked with writing a clear, cohearent step-by-step guide for a user based on the provided reference content and the task.

    The user wants help with the following task:
    "{user_task}"

    Use only the information provided in the reference content below.
    If any step is ambiguous or missing, note that politely rather than guessing.
    """

    reference_text = f"""
    ### Reference Content:
    {reference_text}

    ### Step-by-Step Guide:
    """
    return instructions, reference_text



def create_image_string_descriptors(image_candidates: pd.DataFrame) -> str:
    """
    Creates a string descriptor for each image candidate.

    Args:
        image_candidates (pd.DataFrame): DataFrame containing image candidates.

    Returns:
        list: List of string descriptors for each image candidate.
    """
    image_descriptors = "### Below are the image candidates:\n\n"
    for _, row in image_candidates.iterrows():
        image_id = row["IMAGE_ID"]
        image_path = row["IMAGE_PATH"]
        description = row["DESCRIPTION"]
        page_number = row["PAGE"]
        # image_position = f"X1: {row["IMAGE_X1"]} Y1: {row["IMAGE_Y1"]} X2: {row["IMAGE_X2"]} Y2: {row["IMAGE_Y2"]}"

        image_descriptors += f"IMAGE_ID: {image_id}, PATH: {image_path}, \n Description:\n {description} \n"
    
    return image_descriptors

