import pandas as pd
import time
from datetime import datetime

# New imports
import sys
sys.path.append(".")  
sys.path.append("..\\.")  
sys.path.append("..\\..\\.") 
from src.utils.open_ai_utils import generate_promt_for_openai_api
from src.utils.open_ai_utils import extract_json_from_open_ai_llm_output

from src.ingestion.llm_functions.cortex_llm_functions import vector_embedding_cosine_similarity_search
from src.ingestion.llm_functions.open_ai_llm_functions import extract_task_and_machine_name

from src.rag.retriever import find_document_by_machine_name, narrow_down_relevant_chunks
from src.rag.generator import add_image_references_to_guide
from src.rag.generator import create_step_by_step_prompt
from src.utils.utils import log



def main_RAG_pipeline(user_query: str, machine_name: str = "N/A" , verbose:int = 1) -> str:
    log("Starting RAG pipeline...", level=0)
    log(f"User query: {user_query}", level=1)
    
    task, machine_name = extract_task_and_machine_name(user_query, machine_name)
    log(f"Extracted machine name: {machine_name}", level=1)
    log(f"Extracted task: {task}", level=1)

    log("Finding document ID in snowflake database...", level=1)
    document_info = find_document_by_machine_name(machine_name)
    log(f"Document ID: {document_info['DOCUMENT_ID']}", level=1)
    log(f"Document Name: {document_info['DOCUMENT_NAME']}", level=1)

    log("Calling for Response 2: Finding most relevant chunks of data to solve the task...", level=1)
    task_chunk_df

    # Filtering the task_chunk_df to only include chunks related to the found document
    log(f"Pre - Filtered task chunk dataframe: {len(task_chunk_df)}", level=1)
    filtered_task_chunk_df = narrow_down_relevant_chunks(task_chunk_df, document_info)
    log(f"Post - Filtered task chunk dataframe: {len(filtered_task_chunk_df)}", level=1)

    filtered_task_chunk_df.to_excel("filtered_task_chunk_df.xlsx", index=False)

    # Retrieving a step by step response from the LLM using the relevant chunks
    instructions_3, reference_text_3 = create_step_by_step_prompt(filtered_task_chunk_df, task)
    log("Constructing instructions and reference text for Response 3 step by step guide using the relevant chunks...", level=1)
    log("Reference text:", level=2)
    log(reference_text_3, level=2)
    log("Instructions:", level=2)
    log(instructions_3, level=2)
    log("Calling OpenAI API for Response 3...", level=1)
    response_3 = generate_promt_for_openai_api(
        instructions=instructions_3, 
        input_text = reference_text_3
        ).output_text

    log("Response 3:", level=2)
    log(response_3, level=2)

    # Adding image references to the appropriate steps in the guide
    log("Calling for Response 4: Adding image references to the guide...", level=1)
    response_4 = add_image_references_to_guide(response_3, filtered_task_chunk_df)
    log("Response 4:", level=1)
    log(response_4, level=1)

    return response_4


if __name__ == "__main__":

    machine_name = "WGA1420SIN"
    user_query = f"There is often detergent residues on the laundry when i do a fine wash cycle. My washing machine model is {machine_name}. How can I fix this?" 

    # machine_name = "WAN28282GC"
    # user_query = f"My washing machine model is {machine_name}, and it's often making loud noises while washing. How can I fix this?" 

    main_RAG_pipeline(user_query, machine_name)

 