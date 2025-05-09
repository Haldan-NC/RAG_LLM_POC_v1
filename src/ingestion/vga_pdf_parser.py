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

import sys
sys.path.append(".")  
sys.path.append("..\\.")  
sys.path.append("..\\..\\.") 

from src.utils.utils import get_connection_config, log
from src.utils.utils import SuppressStderr
from src.ingestion.image_extractor import extract_images_from_page


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
        # elif text == 'Error on\nASDS':
        #     text = ''

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


def table_of_interest_index(real_page_num: int) -> int:
    """
    This function is used to skip certain pages that are not relevant for the process of which it is found.
    Defaults to 0, but can be set to 1 for specific pages.

    This is another hardcoded bandaid for the specific document.
    """
    if real_page_num in [612, 653, 742, 749]:
        return 1
    return 0


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
        "images": [images_list]
    }
    log(f"   Added step: {step_num}   explanation: {explanation[:30].replace("\n","")}...   images: {len(images_list)}", level = 3)
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
    guides = []
    # SuppressStderr() is added to suppress the warnings from pdfplumber. "CropBox missing from /Page, defaulting to MediaBox"
    with SuppressStderr():
        with pdfplumber.open(file_path) as pdf:
            pdf_document = list(pdf.pages)  # Load all pages into a list

            guide_active = False
            current_guide = None
            # TQDM doesn't work with SuppressStderr(). 
            for page_idx, page in tqdm(enumerate(pdf_document), desc="Processing pages of VGA guide", unit="page", total=len(pdf_document)):
                real_page_num = page_idx + 1

                # Added to show some form of progress during extraction
                if real_page_num% 150 == 0:
                    log(f"  VGA guide extraction progress: {real_page_num}/{len(pdf_document)} pages", level = 1)

                text = page.extract_text() or ""
                tables = page.extract_tables()

                images_list = extract_images_from_page(page = page, page_num = real_page_num)

                # === Check for guide start ===
                if "Guide Name" in text:
                    if current_guide is not None:
                        guides.append(current_guide)
                        log(f"  Extracted VGA guide nr. {len(guides)} from page {current_guide["steps"][1]["page_number"][0]} - {\
                            current_guide["steps"][step_num]["page_number"][-1]}", level = 2)

                    guide_active = True
                    current_guide = {
                        "page_number": real_page_num,
                        "text": text,
                        "steps": {}
                    }

                # === If we're in a guide context and haven't filled in steps yet ===
                if guide_active and current_guide is not None:
                    found_valid_table = False
                    
                    table_of_interest = table_of_interest_index(real_page_num = real_page_num)
                    for table_idx,table in enumerate(tables): # We are only interested in the first table on the page
                        if table_of_interest == table_idx:
                            if table[0][0] == "Error Text:": # skipping past this table.
                                for i,__table in enumerate(tables):
                                    for j,row in enumerate(__table):
                                        for k,item in enumerate(row):
                                            if item != None:
                                                item = string_hardcoded_bandaid(text = item, real_page_num = real_page_num)
                                                if "Step" in item.replace(" ", "") or "Explanation" in item.replace(" ", ""): 
                                                    found_valid_table = True
                                                    if item == "Step": # One example on page 148 is the cause of this logic
                                                        table = __table # Replace the table with the one that has "Step" in it
                                                        table[j][k] = "Step" # Replace the "S tep" with "Step"
                                                    break
                                if not found_valid_table:
                                    log(f"--> Skipping table on page {real_page_num} as it is not a step table", level = 3)
                                    continue
                                
                            for row in table:
                                log(f"\n  --  page {real_page_num}  --  ", level = 3)

                                row = [string_hardcoded_bandaid(text = x, real_page_num = real_page_num) for x in row if 
                                        string_hardcoded_bandaid(text = x, real_page_num = real_page_num) not in [None,"",'']]
                                        
                                if "Step" in row and "Explanation" in row:
                                    log(f"--> Found step table on page {real_page_num}", level = 3)
                                    found_valid_table = True
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
                                else: 
                                    log("---> No valid step or explanation found", level = 3)
            
        return guides


if __name__ == "__main__":
    # Example usage
    file_path = "data\\Vestas_RTP\\Documents\\VGA_guides\\No communication Rtop - V105 V112 V117 V126 V136 3,3-4,2MW MK3.pdf"
    extract_vga_guide(file_path = file_path)