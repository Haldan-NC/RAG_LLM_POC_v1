import pandas as pd
import fitz
from langchain_community.document_loaders import PDFPlumberLoader
import pdfplumber
import re
import os
import sys
import json
from PIL import Image, ImageDraw
from PIL import UnidentifiedImageError
import io
from tqdm import tqdm
import pickle

import sys
sys.path.append(".")  
sys.path.append("..\\.")  
sys.path.append("..\\..\\.") 

from src.utils.utils import get_connection_config, log
from src.utils.utils import SuppressStderr, check_cache_exists
from src.ingestion.image_extractor import extract_images_from_page
from src.db.db_functions import prepare_documents_df, write_to_table, get_documents_table
from src.db.db_functions import create_vga_guide_table, create_vga_guide_steps_table, create_vga_guide_substeps_table
from src.db.db_functions import create_wind_turbine_table, create_link_table__guide_id__turbine_id
from src.db.db_functions import create_link_table__step_id__dms_no
from src.db.db_functions import get_table


def string_hardcoded_bandaid(text: str, real_page_num: int) -> str:
    """
    Lord forgive me for these hardcoded sins...
    Some extracted text causes issues in the logic of the code. 
    In order avoid excessive logic, hardcoded replacements are made here, suitable for the specific VGA document. 

    The text extractions are flawed when using:
        - Python 3.12.4
        - pdfplumber==0.11.6  (not imported from langchain_community, but import pdfplumber directly)
    """
    if real_page_num == 148:
        if text == '1S. oCftownanreec tI O:\nt o MC with\nTEororolkri ta acnkdn ow\ncAhuetcok wrong\ns ignal in I/O\nl is t from':
            text = '1. Connect to MC with Toolkit and check wrong signal in I/O list from ABB module'
        if text == 'S tep':
            text = 'Step'

    elif real_page_num == 149:
        if text == 'ABB\nmodule.':
            text = ''
        elif text == 'ABB':
            text = ''

    elif real_page_num == 170:
        if text == 'Error on\nASDS':
            text = ''

    elif real_page_num == 253:
        if text == 'rror Text:':
            text = ''

    elif real_page_num == 281:
        if text == '630-16-\nW5 cable':
            text = ''

    elif real_page_num == 482:
        if text == 'sensor\ncable':
            text = ''

    elif real_page_num == 608:
        if text == 'the\nCT8200':
            text = ''

    elif real_page_num == 680:
        if text == '7. Check\nK18 relay\ncoil supply':
            text += '\nand working\nfunction'

    elif real_page_num == 681:
        if text == 'and working\nfunction':
            text = ''

    elif real_page_num == 769:
        if text == '7. Inspect &\ncorrect 130-':
            text += '\nQ1 circuit\nconnection'

    elif real_page_num == 770:
        if text == 'Q1 circuit\nconnection':
            text = ''

    return text


def exception_switch(real_page_num: int) -> bool:
    """
    This function is used to skip certain pages that are not relevant for the process of which it is found.
    Defaults to True, but can be set to False for specific pages.

    This is another hardcoded bandaid for the specific document.
    """
    if real_page_num in [253]:
        return True
    return False


def table_of_interest_index(real_page_num: int, table_of_interest: int) -> int:
    """
    This function is used to define the table index on pages where the guid step and explanation table is on the same page as the overview table.
    
    The variable table_of_interest should be either 0 or 1, depneding on whether the guide overview description
    is on the page. If it is, then the table_of_interest should be 1, otherwise it should be 0.
    The above mentioned naturally assumes that each guide overview:
        1. is extracted as a table
        2. does not contain subtables

    In the cases below, the above mentioned assumptions are not met, and the table_of_interest is altered.

    """
    # Add pages here IFF:
    # 1. The guide overview is on the page
    # 2. The table of interest IS on the page
    if real_page_num in [12]:
        # 3. The guide overview table is not recognized as a table
        return 0

    if real_page_num in [148]:
        # 3. The guide overview contains sub tables, making the table of interest index 3.
        return 3

    # 2. The table of interest IS NOT on the page
    if real_page_num in [277]: 
        # 3. There are more than 1 tables extracted on the page
        return 99

    return table_of_interest


def extract_step_and_explanation_from_row(row: list) -> tuple:
    """
    This function is used to extract the step and explanation from a row of a table.
    Args:
        row (list): The row of the table containing the step and explanation.
    """
    step_label = ""
    explanation = ""

    for item in row:
        if len(row) == 2 and item not in ['',"", None] and step_label == "":
            step_label = item.strip()
            
        elif item not in ['',"", None] and explanation == "":
            explanation = item.strip()
            break

    return step_label, explanation


def get_step_num_and_label(step_label: str, real_page_num: int) -> int:
    """
    In some instances the extracted was incorrect, or the actual text had a type (ref: page 253 where the text was "rror Text:").
    The function tidies up the text using string_hardcoded_bandaid() and returns the step number and label to be used later in the code.
    Args:
        step_label (str): The step label in the left column of the document.
        real_page_num (int): The real page number of the document (not 0 indexed).
    """
    log("---> Found new step and explanation", level = 3)
    step_label = string_hardcoded_bandaid(text = step_label, real_page_num = real_page_num)
    if type(eval(step_label.split(".")[0]))==int:
        step_num = int(step_label.split(".")[0]) 
    else:
        raise ValueError(f"Step label is not an integer: {step_label}")
    return step_label, step_num


def add_step_and_explanation_to_guide(current_guide: dict, step_label: str, step_num: int, explanation: str, 
                                        real_page_num: int, images_list: list) -> dict:
    """
    This function is used to add a step and explanation to the current guide.
    Args:
        current_guide (dict): The current guide to which the step and explanation will be added.
        step_num (int): The step number to be added.
        explanation (str): The explanation text for the step.
        real_page_num (int): The real page number of the document.
        images_list (list): List of images associated with the step.
    """

    log(f"---> Adding step and explanation on page: {real_page_num}", level = 3)
    current_guide["steps"][step_num] = {
        "step_label": step_label.replace("\n", " "),
        "explanation": [explanation.replace("\n", " \n ")],
        "page_number": [real_page_num],
        "images": [images_list],
        "DMS No.": []
    }
    log(f"   Added step: {step_num}   explanation: {explanation[:30].replace("\n","")}...   images: {len(images_list)}", level = 3)
    return current_guide


def extract_dms_no_from_table(table: list) -> dict:
    """
    This function is used to extract the DMS No. from a table.
    Args:
        table (list): The table containing the DMS No.

    """
    extract_rows = False
    dms_no_list = []
    for i,row in enumerate(table):
        if extract_rows:
            item_num = len(dms_no_list)
            dms_no_list.append({
                "DMS No.": row[idx_dms_no],
                "Description": row[idx_description]
            })

        if "DMS No." in row and "Description" in row:
            extract_rows = True
            for j,item in enumerate(row):
                if item == None: 
                    continue
                elif "DMS No." in item:
                    idx_dms_no = j
                elif "Description" in item:
                    idx_description = j

    return dms_no_list


def extract_links_from_page(real_page_num: int, file_path: str, dms_no_list: list) -> list:
    """
    Extracts hyperlinks from a PDF page and returns a new dms_no_list with hyperlinks.
    Args:
        real_page_num (int): The page number of the PDF document.
        file_path (str): The path to the PDF file.
    """
    doc = fitz.open(file_path)
    results = []
    page = doc[real_page_num-1] # it's 0 indexed
    links = page.get_links()
    text_dms_list = [dms_no["DMS No."] for dms_no in dms_no_list]
    
    for link in links:
        uri = link.get("uri")
        rect = link.get("from", None)
        text = page.get_textbox(rect).strip()
        if uri != None: 
            if text in text_dms_list:
                link_details = {
                    "text":  text,
                    "hyperlink": uri
                }
                results.append(link_details)
    
    # Ensured that the hyperlinks are in the same order as the DMS No. list.
    new_results = []
    for i, dms_no in enumerate(dms_no_list):
        for j, link in enumerate(results):
            if dms_no["DMS No."] == link["text"]:
                new_results.append({
                    "DMS No.": dms_no["DMS No."],
                    "Description": dms_no["Description"],
                    "hyperlink": link["hyperlink"]
                })
                break

    return new_results


def add_dms_no_to_current_guide(current_guide: dict, dms_no_list: list, real_page_number: int) -> dict:
    """
    This function adds the DMS No. to the current guide. 
    
    Known issue 1:
    The given list of tables which is iterated over does not nest the DMS No. table.
    What that means, is that if two DMS No. tables are on the same page, the function will add both the DMS No. tables to both steps.
    Page 77 (step 22 and 23) is an example of this.

    Known issue 2:
    If a DMS No. table spans 2 pages, the function will only add the DMS No. table on the first page. An example of this is page 273-274 (step 6).

    Args:
        current_guide (dict): The current guide to which the DMS No. will be added.
        dms_no_list (list): The list of DMS No. to be added.
        real_page_number (int): The real page number of the document.
        hyper_links (list): The list of hyperlinks associated with the DMS No. table.

    Returns:
        current_guide (dict): The current guide with the DMS No. added.
    """
    for step in current_guide["steps"]:
        for page_idx, step_page in enumerate(current_guide["steps"][step]["page_number"]):
            if step_page == real_page_number:
                if "Relevant documentation" in current_guide["steps"][step]["explanation"][page_idx]:
                    current_guide["steps"][step]["DMS No."] += dms_no_list
                    log(f"---> Added DMS No. to step: {step}  on page: {real_page_number}   DMS No:{dms_no_list}", level = 3)
                    break

    return current_guide


def add_explanation_to_guide(current_guide: dict, step_num: int, explanation: str, real_page_num: int, images_list: list) -> dict:
    """
    This function is used to add an explanation to the current guide.
    Args:
        current_guide (dict): The current guide to which the explanation will be added.
        step_num (int): The step number to which the explanation will be added.
        explanation (str): The explanation text for the step.
        real_page_num (int): The real page number of the document.
        images_list (list): List of images associated with the step.
    """
    log(f"---> Adding explanation on page: {real_page_num}", level = 3)
    current_guide["steps"][step_num]["explanation"].append(explanation.replace("\n", " \n "))
    current_guide["steps"][step_num]["page_number"].append(real_page_num)
    current_guide["steps"][step_num]["images"].append(images_list)
    log(f"   Added sub step to step: {step_num}   explanation: {explanation[:30].replace("\n","")}...   images: {len(images_list)}", level = 3)

    return current_guide




def extract_vga_guide(file_path: str) -> list:
    """
    Extracts the contant from the VGA guide and returns a dataframe with the content. 
    
    The function uses pdfplumber to extract text and tables from the PDF file.
    It processes the text to identify guides, steps, explanations, images, and organizes them into a structured format.

    The code is not clean nor general, but works for the specific document.
    It is a work in progress and will be improved in the future.

    Args:
        file_path (str): Path to the PDF file.
        manual_id (int): Manual ID for the document.
    
    Returns:
        pd.DataFrame: DataFrame containing the extracted content.
    """

    if check_cache_exists(file_path = "data\\Vestas_RTP\\Cache\\vga_guides.pickle"):
        return load_cached_guides()
        
    else:
        guides = []
        # SuppressStderr() is added to suppress the warnings from pdfplumber. "CropBox missing from /Page, defaulting to MediaBox". Also supresses other standard output.
        with SuppressStderr():
            with pdfplumber.open(file_path) as pdf:
                pdf_document = list(pdf.pages)  # Load all pages into a list

                guide_active = False
                current_guide = None
                table_of_interest = 0

                for page_idx, page in enumerate(pdf_document):
                    real_page_num = page_idx + 1
                    step_and_expl_extracted = False

                    # Added to terminal output to show progress of the extraction.
                    if (real_page_num% 150 == 0) or (real_page_num == len(pdf_document)):
                        log(f"  VGA guide extraction progress: {real_page_num}/{len(pdf_document)} pages", level = 1)

                    text = page.extract_text() or ""
                    tables = page.extract_tables()

                    images_list = extract_images_from_page(page = page, page_num = real_page_num, image_path = file_path)

                    # Check for guide start. The only thing that all guide overview pages has is common is the "Guide Name" text.
                    if "Guide Name" in text:
                        if current_guide is not None:
                            guides.append(current_guide)
                            log(f"  Extracted VGA guide nr. {len(guides)} from page {current_guide["steps"][1]["page_number"][0]} - {\
                                current_guide["steps"][step_num]["page_number"][-1]}", level = 2)

                        table_of_interest = 1
                        current_guide = {
                            "page_number": real_page_num,
                            "text": text,
                            "steps": {}
                        }
                    
                    else:
                        table_of_interest = 0

                    # The Table of interest index is used on 3 exceptions, where the rule based flow of the guide overview page is not working as expected.
                    # More info in the doc string.
                    table_of_interest = table_of_interest_index(real_page_num = real_page_num, 
                                                                table_of_interest = table_of_interest)

                    for table_idx, table in enumerate(tables): # We are only interested in the first table on the page
                        if step_and_expl_extracted:
                            # Any tables after the table with the step and explanation are sub tables.
                            # The add_dms_no_to_current_guide function is used to add the DMS No. to the current guide.
                            dms_no_list = extract_dms_no_from_table(table = table)
                            dms_no_list = extract_links_from_page(real_page_num = real_page_num, file_path = file_path, dms_no_list= dms_no_list)
                            current_guide = add_dms_no_to_current_guide(
                                current_guide = current_guide,
                                dms_no_list = dms_no_list,
                                real_page_number = real_page_num
                            )

                        if table_of_interest == table_idx:
                            for row in table:
                                log(f"\n  --  page {real_page_num}  --  ", level = 3)

                                row = [string_hardcoded_bandaid(text = x, real_page_num = real_page_num) for x in row if 
                                        string_hardcoded_bandaid(text = x, real_page_num = real_page_num) not in [None,"",'']]
                                        
                                if "Step" in row and "Explanation" in row:
                                    log(f"--> Found step table on page {real_page_num}", level = 3)
                                    step_and_expl_extracted = True
                                    # Continue to the next row, to avoid writing "Step" and "Explanation" to the dictionary.
                                    continue

                                step_label, explanation = extract_step_and_explanation_from_row(row = row)
                                        
                                if step_label in ['',"", None] and (explanation not in ['',"", None] or len(images_list)> 0):
                                    if not exception_switch(real_page_num): # Defaults to False, unless an exception is made.
                                        current_guide = add_explanation_to_guide(
                                            current_guide = current_guide,
                                            step_num = step_num,
                                            explanation = explanation,
                                            real_page_num = real_page_num,
                                            images_list = images_list
                                        )
                                        step_and_expl_extracted = True

                                elif step_label not in ['',"", None] and explanation not in ['',"", None]:
                                    if not exception_switch(real_page_num): # Defaults to False, unless an exception is made.
                                        step_label, step_num = get_step_num_and_label(step_label = step_label,
                                                                            real_page_num = real_page_num
                                        )   

                                        current_guide = add_step_and_explanation_to_guide(
                                            current_guide = current_guide,
                                            step_label = step_label,
                                            step_num = step_num,
                                            explanation = explanation,
                                            real_page_num = real_page_num,
                                            images_list = images_list
                                        )
                                    step_and_expl_extracted = True

                                else: 
                                    log("---> No valid step or explanation found", level = 3)
            
            log(f"  Extracted {len(guides)} guides from the VGA guide", level = 1)
            cache_guides(guides)

            return guides


def expand_mk_range(mk_str: str) -> list:
    """
    Expands a string representing a range of MK numbers into a list of individual MK numbers. 
    Is used to define the wind turbine models for each guide.
    """
    match = re.match(r'MK(\d+)-(\d+)', mk_str)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        return [f'MK{i}' for i in range(start, end+1)]
    else:
        return [mk_str]


def normalize_generator(g: str) -> str:
    """
    There are inconsistencies with the use of , and . in the machine model string. 
    This is likely due to a max of danish and english speaking people creating the document.
    """
    return g.replace(',', '.').strip()


def extract_machines(s: str) -> set:
    """
    Extracts the machine models from the given string.
    The string is expected to contain segments that look like: "V105 V112 ... 3-4,2MW MK0-3".
    The function uses regex and black magic to find the segments and extract the machine models.
    """
    # Split string into segments ending in a generation spec
    # Each segment should look like: "V105 V112 ... 3-4,2MW MK0-3"
    segments = re.findall(r'(?:V\d+\s+)+(?:[\d.,/-]+MW)\s+MK[\d\-]+', s)
    machines = set()

    for segment in segments:
        frames = re.findall(r'\bV\d+\b', segment)
        generator = re.search(r'[\d.,/-]+MW', segment)
        mk_match = re.search(r'MK[\d\-]+', segment)

        if not (frames and generator and mk_match):
            continue

        generator_str = normalize_generator(generator.group())
        generations = expand_mk_range(mk_match.group())

        for frame in frames:
            for mk in generations:
                machines.add(f"{frame} {generator_str} {mk}")

    return sorted(machines)


def append_vga_guide_to_documents_table() -> pd.DataFrame:
    """
    Adds the VGA guide to the documents table in Snowflake.
    This is seperate from the main ingenstion process as the VGA guide is a special case in a seperate directory.

    """

    pdf_files_path = "data\\Vestas_RTP\\Documents\\VGA_guides\\"
    documents_df = prepare_documents_df(pdf_files_path = pdf_files_path)
    write_to_table(df = documents_df, table_name="DOCUMENTS")
    documents_df = get_documents_table()

    return documents_df


def extract_guide_name_from_text(text: str) -> str:
    """
    Extracts the guide name from the text.
    The guide name is the first line of the text.
    """
    return text.replace("Guide Name: ", "").strip().split("\n")[0]


def create_vga_guide_dataframe(guides: list, document_id: int) -> pd.DataFrame:
    """
    A function which creates the dataframe which is written to the database.
    Each row corresponds to a guide in the VGA pdf.

    The dataframe contains the following columns:
        - DOCUMENT_ID: The document ID of the guide (foreign key)
        - GUIDE_NUMBER: The number of the guide
        - PAGE_NUMBER: The page number of the guide
        - GUIDE_NAME: The name of the guide
        - STEPS: The number of steps in the guide
        - TURBINE_MODELS: The turbine models in the guide
    """
    guides_df = pd.DataFrame(guides)
    guides_df["steps"] = [len(guides[i]['steps']) for i in range(len(guides))]
    guides_df.reset_index(inplace=True)
    guides_df.rename(columns={"text": "GUIDE_NAME", "page_number": "PAGE_NUMBER", "steps":"STEPS", "index": "GUIDE_NUMBER"}, inplace=True)
    guides_df["GUIDE_NAME"] = [extract_guide_name_from_text(text) for text in guides_df["GUIDE_NAME"]]
    guides_df["TURBINE_MODELS"] = ""

    all_machines = set()
    for i, example in enumerate(guides_df["GUIDE_NAME"].copy()):
        machines = extract_machines(example)
        for m in machines:
            all_machines.add(m)
        guides_df.at[i, "TURBINE_MODELS"] = " | ".join(machines)
    guides_df["DOCUMENT_ID"] = document_id
    return guides_df


def find_guide_ID_from_name(guides_df: pd.DataFrame, guide_name: str) -> int:
    """
    Finds the guide ID from the guide name.
    The guide ID is used to reference the guide in the database.
    """
    for i, row in guides_df.iterrows():
        if row["GUIDE_NAME"] == guide_name:
            return row["GUIDE_ID"]
    return None


def create_vga_guide_steps_dataframe(guides: list, guides_df: pd.DataFrame) -> pd.DataFrame:
    """
    A function which creates the dataframe which is written to the database.
    Each row corresponds to an entire step in a guide in the VGA pdf.

    The dataframe contains the following columns:
        - DOCUMENT_ID: The document ID of the guide (foreign key)
        - GUIDE_ID: The guide ID of the guide (foreign key)
        - GUIDE_NUMBER: The number of the guide (slightly redundant in terms of normalizing the database)
        - PAGE_START: The start page of the step
        - PAGE_END: The end page of the step
        - STEP: The step number
        - STEP_LABEL: The step label
        - TEXT: The concatenated text of the entire step, whether its a small step or a 4 page step.
    """
    combined_steps_rows = []
    document_id = guides_df["DOCUMENT_ID"].iloc[0]

    for g_idx,guide in enumerate(guides):
        guide_id = find_guide_ID_from_name(guides_df = guides_df, 
                guide_name = extract_guide_name_from_text(guide["text"]))

        row_dict = {
                "DOCUMENT_ID": document_id,
                "GUIDE_ID": guide_id,
                "GUIDE_NUMBER": g_idx,
                "PAGE_START": guide["page_number"],
                "PAGE_END": guide["page_number"],
                "STEP": 0,
                "STEP_LABEL": "Guide Overview",
                "TEXT": guide['text']
            }
        # Adding the overview text as a row as it might contain useful information
        combined_steps_rows.append(row_dict)
        
        for s_idx, step in enumerate(guide["steps"]):
            combined_explanation = ""
            page_end = 0
            for ex_idx, explanation in enumerate(guide["steps"][step]["explanation"]):
                combined_explanation += explanation + " "
                if ex_idx == 0:
                    page_start = 0
                elif ex_idx == len(guide["steps"][step]["explanation"]) - 1:
                    page_end = guide["steps"][step]["page_number"][ex_idx]

            row_dict = {
                "DOCUMENT_ID": document_id,
                "GUIDE_ID": guide_id,
                "GUIDE_NUMBER": g_idx,
                "PAGE_START": guide["steps"][step]["page_number"][0],
                "PAGE_END": guide["steps"][step]["page_number"][ex_idx],
                "STEP": step,
                "STEP_LABEL": guide["steps"][step]["step_label"],
                "TEXT": guide["steps"][step]["explanation"][ex_idx]
            }
            # Adding each Step and Explaination as a row
            combined_steps_rows.append(row_dict)

    combined_steps_df = pd.DataFrame(combined_steps_rows)
    return combined_steps_df
    

def create_vga_guide_substeps_dataframe(guides: list, guides_df: pd.DataFrame) -> pd.DataFrame:
    """ 
    A function which creates the dataframe which is written to the database.
    Each row corresponds to a sub step in a step in the VGA pdf.

    A sub step is defined by a section of a step which covers multiples pages, which happens more often than not in the VGA guide.
    A sub step is at most a single page.
    Example: Page 278-280 (step 2.) covers 3 pages, thus, contains 3 sub steps.

    The dataframe contains the following columns:
        - DOCUMENT_ID: The document ID of the guide (foreign key)
        - GUIDE_ID: The guide ID of the guide (foreign key)
        - GUIDE_NUMBER: The number of the guide (slightly redundant in terms of normalizing the database)
        - PAGE_NUMBER: The page number of the step
        - STEP: The step number
        - STEP_LABEL: The step label
        - TEXT: The text of the specific sub step. Sub steps occur when a single step covers multiple pages. 
    """
    substeps_rows = []
    document_id = guides_df["DOCUMENT_ID"].iloc[0]

    for g_idx,guide in enumerate(guides):

        guide_id = find_guide_ID_from_name(guides_df = guides_df, 
                guide_name = extract_guide_name_from_text(guide["text"]))

        row_dict = {
                "DOCUMENT_ID": document_id,
                "GUIDE_ID": guide_id,
                "GUIDE_NUMBER": g_idx,
                "PAGE_NUMBER": guide["page_number"],
                "STEP": 0,
                "STEP_LABEL": "Guide Overview",
                "TEXT": guide['text'],
            }
        # Adding the overview text as a row as it might contain useful information
        substeps_rows.append(row_dict)
        
        for s_idx, step in enumerate(guide["steps"]):
            for ex_idx, explanation in enumerate(guide["steps"][step]["explanation"]):
                row_dict = {
                    "DOCUMENT_ID": document_id,
                    "GUIDE_ID": guide_id,
                    "GUIDE_NUMBER": g_idx,
                    "PAGE_NUMBER": guide["steps"][step]["page_number"][ex_idx],
                    "STEP": step,
                    "STEP_LABEL": guide["steps"][step]["step_label"],
                    "TEXT": guide["steps"][step]["explanation"][ex_idx],
                    # "IMAGES": guide["steps"][step]["images"][ex_idx],
                }
                # Adding each Step and Explaination as a row
                substeps_rows.append(row_dict)

    substeps_df = pd.DataFrame(substeps_rows)
    return substeps_df
    

def create_wind_turbine_dataframe(guides_df: pd.DataFrame) -> pd.DataFrame:
    """
    A function which creates the dataframe which is written to the database.
    Each row corresponds to a wind turbine model.

    The dataframe contains the following columns:
        - WIND_TURBINE_ID: The ID of the wind turbine model
        - WIND_TURBINE_NAME: The name of the wind turbine model
        - SIZE: The size of the wind turbine model (V105, V112, V117, V126, V136)
        - POWER: The generation of the wind turbine model (3.3-4.2MW)
        - MK_VERSION: The generation of the wind turbine model (3.3-4.2MW)
    """
    names = []
    sizes = []
    powers = []
    mk_versions = []
    for idx, row in guides_df.iterrows():
        row_string = guides_df.at[idx, "TURBINE_MODELS"]
        row_list = row_string.split(" | ")
        for model in row_list:
            if model == "":
                continue
            size = model.split(" ")[0]
            power = model.split(" ")[1]
            mk_version = model.split(" ")[2]
            names.append(model)
            sizes.append(size)
            powers.append(power)
            mk_versions.append(mk_version)

    wind_turbine_df = pd.DataFrame({
        "TURBINE_NAME": names,
        "SIZE": sizes,
        "POWER": powers,
        "MK_VERSION": mk_versions
    })
    
    wind_turbine_df.drop_duplicates(subset=["TURBINE_NAME"], inplace=True)
    wind_turbine_df.reset_index(drop=True, inplace=True)

    return wind_turbine_df


def create_link_dataframe__guide_id__turbine_id()-> pd.DataFrame:
    """
    A function which creates a linking dataframe which is written to the database.
    Each row corresponds to a link between a guide from VGA_GUIDES and a wind turbine model from WIND_TURBINES.
    The dataframe contains the following columns:
        - GUIDE_ID: The ID of the guide (foreign key)
        - TURBINE_ID: The ID of the wind turbine model (foreign key)
    """
    turbine_df = get_table(table_name = "WIND_TURBINES")
    guides_df = get_table(table_name = "VGA_GUIDES")

    rows = []
    for i, row in guides_df.iterrows():
        turbine_models = row["TURBINE_MODELS"].split(" | ")
        for model in turbine_models:
            if model == "":
                continue
            turbine_id = turbine_df[turbine_df["TURBINE_NAME"] == model]["TURBINE_ID"].values[0]
            rows.append({
                "GUIDE_ID": row["GUIDE_ID"],
                "TURBINE_ID": turbine_id
            })

    link_df = pd.DataFrame(rows)
    return link_df

    
def create_link_dataframe__step_id__dms_no(guides: dict )-> pd.DataFrame:
    """
    A function which creates a linking dataframe which is written to the database.
    Each row corresponds to a link between a step from VGA_GUIDE_STEPS and a DMS No. from DMS_NO.
    The dataframe contains the following columns:
        - STEP_ID: The ID of the step (foreign key)
        - DMS_NO: The DMS No. (potential key to Documents table)
    """
    steps_df = get_table(table_name = "VGA_GUIDE_STEPS")

    rows = []
    for i, row in steps_df.iterrows():
        guide_id = row["GUIDE_ID"]  # used for the database relation
        guide_number = row["GUIDE_NUMBER"] # used for indexing in the guides dict
        step_number = row["STEP"] # use for indexing in the guides dict
        if step_number == 0:
            continue
        dms_no_list = guides[guide_number]["steps"][step_number]["DMS No."] # a list of dicts
        for dms_no_entry in dms_no_list:
            rows.append({
                "GUIDE_ID": row["GUIDE_ID"], # Might be useful for extracting the DMS_NO pr. guide.
                "GUIDE_STEP_ID": row["GUIDE_STEP_ID"],
                "DMS_NO": dms_no_entry["DMS No."],
                "DESCRIPTION": dms_no_entry["Description"],
                "HYPERLINK": dms_no_entry["hyperlink"]
            })

    link_df = pd.DataFrame(rows)
    link_df.drop_duplicates(subset=["GUIDE_STEP_ID", "DMS_NO"], inplace=True)
    link_df.reset_index(drop=True, inplace=True)

    return link_df


def cache_guides(guides: dict):
    log(f"Caching parsed guides dict to  data/Vestas_RTP/Cache", level = 1)
    with open("data\\Vestas_RTP\\Cache\\vga_guides.pickle", "wb") as file:
        pickle.dump(guides, file)


def load_cached_guides():
    with open("data\\Vestas_RTP\\Cache\\vga_guides.pickle", "rb") as file:
        guides = pickle.load(file)
    log(f"Loaded guides dict from cache", level = 1)
    return guides


if __name__ == "__main__":
    
    file_path = "data\\Vestas_RTP\\Documents\\VGA_guides\\No communication Rtop - V105 V112 V117 V126 V136 3,3-4,2MW MK3.pdf"
    guides = extract_vga_guide(file_path)

    # guides_df = get_table(table_name = "VGA_GUIDES")
    # print("guides_df:")
    # print(guides_df.head())
    # print("\n")

    # turbine_df = create_wind_turbine_dataframe(guides_df = guides_df)
    # create_wind_turbine_table()
    # write_to_table(df = turbine_df, table_name = "WIND_TURBINES")

    # turbine_link_df = create_link_dataframe__guide_id__turbine_id()
    # create_link_table__guide_id__turbine_id()
    # write_to_table(df = turbine_link_df, table_name = "LINK_GUIDE_TURBINE")

    # dms_no_link_df = create_link_dataframe__step_id__dms_no(guides = guides)
    # create_link_table__step_id__dms_no()
    # write_to_table(df = dms_no_link_df, table_name = "LINK_STEP_DMS")

    # steps_df = create_vga_guide_steps_dataframe(guides = guides, guides_df = guides_df)
    # print("steps_df:")
    # print(steps_df.head())
    # print("\n")
    
    # substeps_df = create_vga_guide_substeps_dataframe(guides = guides, guides_df = guides_df)
    # print("substeps_df:")
    # print(substeps_df.head())
    # print("\n")


