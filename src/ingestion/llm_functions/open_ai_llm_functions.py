import os 
import time
import sys
sys.path.append(".")  
sys.path.append("..\\.")  
sys.path.append("..\\..\\.") 
sys.path.append("..\\..\\..\\.") 
import pandas as pd
import base64
from langchain.evaluation import load_evaluator

from src.utils.open_ai_utils import get_openai_api_key, generate_promt_for_openai_api
from src.utils.open_ai_utils import extract_json_from_open_ai_llm_output
from src.utils.open_ai_utils import get_openai_client
from src.utils.utils import log, log_execution_time


def extract_task_and_machine_name(user_query: str, machine_name: str = "N/A") -> tuple:
    log("Calling for Response 1: Extracting machine name and task...", level=1)
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

    return task, machine_name


def create_rows_from_TOC_dict(toc_dict: dict, parent_section: str = None) -> list:
    """
    This function takes a dictionary representing a table of contents and recursively traverses it to create a flat list of dictionaries.
    Each dictionary in the list represents a section of the table of contents, including its name, number, page, and parent section number.

    Args:
        toc_dict (dict): a dictionary returned from extract_TOC_OpenAI()
        parent_section (str, optional): Should be none when the function is called, as it is used for recursion. Defaults to None.

    Returns:
        toc_df (pd.DataFrame): a dataframe containing the sections of the table of contents.
    """
    rows = []

    # Get info from the current toc_dict
    section = toc_dict.get("Section")
    section_number = toc_dict.get("Section Number")
    page = toc_dict.get("Page")

    # The levenshtein distance is used to ensure the the section called "Table of Contents" is not added to the dataframe
    evaluator = load_evaluator("string_distance")
    levenshtein_score_toc = evaluator.evaluate_strings(
    prediction=section,
    reference="Table of Contents",
    metric="levenshtein"
    )["score"]  # This will be a float between 0 and 1, where 0 means identical

    if levenshtein_score_toc > 0.1:  # if the levenshtein distance is very small its likely to match "Table of Contents"
        rows.append({
            "SECTION": section,
            "SECTION_NUMBER": section_number,
            "PAGE": page,
            "PARENT_SECTION_NUMBER": parent_section
        })

    # Recurse into each sub-section, if any
    for subsection in toc_dict.get("Sub Sections", []):
        log(f"Rows: {rows}", level=2)
        rows.extend(create_rows_from_TOC_dict(subsection, parent_section=section_number))

    return rows


def extract_TOC_OpenAI(text: str) -> pd.DataFrame:
    prompt = (
    """
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
    - "Section Number" must be represented as a float - E.G: 1, 2, 5.3, 1,4, etc.
    - "Section Number" is usually before the section name, but not always, infer it it you must.
    - Ignore any text that is not part of the table of contents.
    - Ensure that sub-sections are nested appropriately under their parent section.
    - Page numbers should be extracted as integers, if possible.
    - Be tolerant of inconsistencies in formatting, spacing, or punctuation (e.g. dashes, colons, ellipses).
    - Do not include duplicate or repeated sections.
    - You should only consider items which are part of the table of contents, nothing before, nothing after.
    - "Section" must consist of words
    - You must include a top level key value pair called "Section":"Table of contents".

    """
    f"Text:\n{text}"
    )

    start_time = time.time()
    result = generate_promt_for_openai_api(instructions = prompt, 
                                            input_text = text)

    output_dict = extract_json_from_open_ai_llm_output(result.output_text)
    toc_rows = create_rows_from_TOC_dict(output_dict)
    toc_df = pd.DataFrame(toc_rows)

    # DropnaN values in the dataframe. Sections with no parent sections have None replaced with "N/A".
    toc_df = toc_df.dropna(inplace=True, how='all')

    return toc_df


def call_openai_api_for_image_description(file_path: str, prompt: str) -> str:
    """
    Calls OpenAI API to generate a description for the image using the provided context string.

    Args:
        file_uri (str): URI of the image file.
        context_string (str): Context string for the image.

    Returns:
        str: Generated description for the image.
    """

    start_time = time.time()
    with open(file_path, "rb") as image_file:
        b64_image = base64.b64encode(image_file.read()).decode("utf-8")

    client = get_openai_client()

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text":  prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{b64_image}"}
                ],
            }
        ],
    )

    log_execution_time(time_start =start_time, description="OpenAI API call for (image+text)->text description using gpt-4o-mini", level=1)

    return response.output_text




if __name__ == "__main__":
    pass