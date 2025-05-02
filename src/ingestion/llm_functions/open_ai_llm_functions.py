import os 
import time
import sys
sys.path.append("..\\.")  
sys.path.append("..\\..\\.") 
sys.path.append("..\\..\\..\\.") 
import pandas as pd
from langchain.evaluation import load_evaluator

from utils.open_ai_utils import get_openai_api_key, generate_promt_for_openai_api
from utils.open_ai_utils import extract_json_from_open_ai_llm_output




def create_df_from_TOC_dict(toc_dict: dict, parent_section: str = None) -> list:
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
        rows.extend(create_df_from_TOC_dict(subsection, parent_section=section_number))
    toc_df = pd.DataFrame(rows)

    return toc_df


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
    toc_df = create_df_from_TOC_dict(output_dict)

    return toc_df



if __name__ == "__main__":
    pass