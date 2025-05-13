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

from src.llm_functions.cortex_llm_functions import vector_embedding_cosine_similarity_search
from src.llm_functions.open_ai_llm_functions import extract_task_and_machine_name

from src.rag.retriever import find_document_by_machine_name, narrow_down_relevant_chunks, get_best_document_for_machine
from src.rag.generator import add_image_references_to_guide
from src.rag.generator import create_step_by_step_prompt
from src.utils.utils import log



def main_RAG_pipeline(user_query: str, machine_name: str = "N/A", verbose: int = 1) -> str:
    log("Starting RAG pipeline...", level=0)
    log(f"User query: {user_query}", level=1)
    
    task, machine_name = extract_task_and_machine_name(user_query, machine_name)
    log(f"Extracted machine name: {machine_name}", level=1)
    log(f"Extracted task: {task}", level=1)

    # Use new document identification
    document_info = get_best_document_for_machine(machine_name)
    if not document_info:
        log(f"No matching document found for machine: {machine_name}", level=0)
        return f"Sorry, I couldn't find any documentation for the {machine_name} machine type."
    
    log(f"Found matching document: {document_info['DOCUMENT_NAME']}", level=1)
    log(f"Document ID: {document_info['DOCUMENT_ID']}", level=1)

    # Get relevant chunks
    log("Calling for Response 2: Finding most relevant chunks of data to solve the task...", level=1)
    task_chunk_df = vector_embedding_cosine_similarity_search(
        input_text=task, 
        chunk_size="small", 
        top_k=20, 
        similarity_threshold=0.1
    )

    # Filter chunks to only include those from the matched document
    filtered_task_chunk_df = narrow_down_relevant_chunks(task_chunk_df, document_info)
    log(f"Pre-filtered task chunk dataframe: {len(task_chunk_df)}", level=1)
    log(f"Post-filtered task chunk dataframe: {len(filtered_task_chunk_df)}", level=1)

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
        input_text=reference_text_3
    ).output_text

    log("Response 3:", level=1)
    log(response_3, level=1)
    
    return response_3


    #NOTE: Commenting out images as the step creating image descriptions is not tailored to the task yet.

    # Adding image references to the appropriate steps in the guide
    # log("Calling for Response 4: Adding image references to the guide...", level=1)
    # response_4 = add_image_references_to_guide(response_3, filtered_task_chunk_df)
    # log("Response 4:", level=1)
    # log(response_4, level=1)

    # return response_4


if __name__ == "__main__":

    # 
    # user_query = f"I get an error text stating: HVCBNotHealthy for a V126 wind turbine." 

    # machine_name = "V105"
    # user_query = f"The Smoke Detector System for {machine_name} is not in Maintenance or Operating mode within the StartupTime. How do I fix this?"
 
    machine_name = "V112"
    user_query = f"What does it mean when the alarm is triggered due to the SwitchgearHVCBOk signal changing to false\
                 in the High Voltage Circuit Breaker? The windmill i'm sitting in is {machine_name}."

    main_RAG_pipeline(user_query)

 