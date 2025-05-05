import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keyring
import os 
import snowflake.connector as sf_connector # ( https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-connect)
from snowflake.connector.pandas_tools import write_pandas # (https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-api#write_pandas)
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.evaluation import load_evaluator
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import time
import re
import json

from io import BytesIO
import fitz 
from shapely.geometry import box
from shapely.ops import unary_union
from PIL import Image, ImageDraw
import cv2
from datetime import datetime


# New imports
import sys
sys.path.append(".")  
sys.path.append("..\\.")  
sys.path.append("..\\..\\.") 
from src.utils.utils import get_config
from src.utils.open_ai_utils import get_openai_api_key, generate_promt_for_openai_api
from src.utils.open_ai_utils import extract_json_from_open_ai_llm_output
from src.db.db_functions import get_cursor

from src.ingestion.llm_functions.cortex_llm_functions import vector_embedding_cosine_similarity_search
from src.ingestion.llm_functions.cortex_llm_functions import vector_embedding_cosine_similarity_between_texts
from src.ingestion.llm_functions.open_ai_llm_functions import call_openai_api_for_image_description

from src.rag.retriever import find_document_by_machine_name, narrow_down_relevant_chunks
from src.rag.generator import add_image_references_to_guide
from src.rag.generator import create_image_string_descriptors
from src.rag.generator import create_step_by_step_prompt



def log(message: str) -> None:
    """
    Logs a message to the console. Wrapper function made for easy modification in the future.
    
    Should be moved to utils when i can figure out a good way to define the global verbosity level.
    """
    if VERBOSE > 0:
        cur_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f" {cur_datetime}  [LOG]  {message}")

def log2(message: str) -> None:
    """
    Logs a message to the console. Wrapper function made for easy modification in the future.
    
    Should be moved to utils when i can figure out a good way to define the global verbosity level.
    """
    if VERBOSE == 2:
        cur_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f" {cur_datetime}  [LOG]  {message}")


def main_RAG_pipeline(user_query: str, machine_name: str = "N/A" , verbose:int = 1) -> str:
    global VERBOSE
    VERBOSE = 1

    if verbose:
        log("Verbose mode is ON.")
    else:
        log("Verbose mode is OFF.")
        VERBOSE = 0

    log("Starting RAG pipeline...")
    log(f"User query: {user_query}")
    log("Calling for Response 1: Extracting machine name and task...")
    response_1 = generate_promt_for_openai_api(
        instructions=f"""
        Extract from the following user query:
        1. The machine name or type. Let the key be "machine_name". If the user defined machine name is not "N/A", use that.
        2. A one-sentence description of the task. Let the key be "task".

        User defined machine name: {machine_name}

        Return as JSON.
        User query: 
        """, 
        input_text = user_query
        )

    response_1 = extract_json_from_open_ai_llm_output(response_1.output_text)
    machine_name = response_1['machine_name']
    task = response_1['task']
    log(f"Extracted machine name: {machine_name}")
    log(f"Extracted task: {task}")

    log("Finding document ID in snowflake database...")
    document_info = find_document_by_machine_name(machine_name)

    log(f"Document ID: {document_info['DOCUMENT_ID']}")
    log(f"Document Name: {document_info['DOCUMENT_NAME']}")
    log("Calling for Response 2: Finding most relevant chunks of data to solve the task...")
    task_chunk_df = vector_embedding_cosine_similarity_search(input_text = task, chunk_size = "small")

    # Filtering the task_chunk_df to only include chunks related to the found document
    log(f"Pre - Filtered task chunk dataframe: {len(task_chunk_df)}")
    filtered_task_chunk_df = narrow_down_relevant_chunks(task_chunk_df, document_info)
    log(f"Post - Filtered task chunk dataframe: {len(filtered_task_chunk_df)}")

    # Retrieve a step by step response from the LLM using the relevant chunks
    instructions_3, reference_text_3 = create_step_by_step_prompt(filtered_task_chunk_df, task)

    log("Calling for Response 3: Constructing a step by step guide using the relevant chunks...")
    log2("\nReference text:")
    log2(reference_text_3)
    log2("\nInstructions:")
    log2(instructions_3)
    log("\nCalling OpenAI API for Response 3...")
    response_3 = generate_promt_for_openai_api(
        instructions=instructions_3, 
        input_text = reference_text_3
        ).output_text

    log2("Response 3:")
    log2(response_3)

    log("Calling for Response 4: Adding image references to the guide...")
    response_4 = add_image_references_to_guide(response_3, filtered_task_chunk_df)
    log("Response 4:")
    log(response_4)

    return response_4


if __name__ == "__main__":

    machine_name = "WGA1420SIN"
    user_query = f"There is often detergent residues on the laundry when i do a fine wash cycle. My washing machine model is {machine_name}. How can I fix this?" 

    main_RAG_pipeline(user_query, machine_name)

 