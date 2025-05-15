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





def extract_userguide(file_path: str) -> list:
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
        
    # guides = []
    # # SuppressStderr() is added to suppress the warnings from pdfplumber. "CropBox missing from /Page, defaulting to MediaBox". Also supresses other standard output.
    # with SuppressStderr():
    #     with pdfplumber.open(file_path) as pdf:
    #         pdf_document = list(pdf.pages)  # Load all pages into a list
    #         for page_idx, page in enumerate(pdf_document):
    #             real_page_num = page_idx + 1
    #             step_and_expl_extracted = False

    #             text = page.extract_text()
    #             tables = page.extract_tables()

    #             print("text:")
    #             print(text)
    #             print("\n")
    #             print("tables:")
    #             print(tables)
    #             print("")


    doc = fitz.open(file_path)
    for idx,page in enumerate(doc):
        # Extract text from the page
        real_page_num = idx + 1
        print(f"Page {real_page_num}:")
        text = page.get_text("text")
        print("--- Text from page: --- ")
        print(text)
        print("\n")
        images = page.get_images(full=True)
        print("--- Images from page: ---")
        for img in images:
            xref = img[0]
            images_coords = page.get_image_rects(xref)
            print(f"Image xref: {xref}")
            print(f"Image rects: {images_coords}")
            # base_image = doc.extract_image(xref)
            # image_bytes = base_image["image"]
            # image = Image.open(io.BytesIO(image_bytes))
            # image.show()

        for coords in images_coords:
            x0, y0, x1, y1 = coords
            print(f"Image coordinates: ({x0}, {y0}, {x1}, {y1})")
        

        blocks = page.get_text("dict")["blocks"]
        print("--- Blocks from page: ---")
        for b in blocks:
            if "lines" in b:
                for line in b["lines"]:
                    row = []
                    for span in line["spans"]:
                        row.append(span["text"])
                    print("\t".join(row))  # Approximate table row
        print("\n" + "-"*80 + "\n")
        print("")


        # blocks = page.get_text("dict")["blocks"]
        # for block in blocks:
        #     if "lines" in block:
        #         for line in block["lines"]:
        #             for span in line["spans"]:
        #                 print(span["text"], span["bbox"])  # You can compute gaps here

if __name__ == "__main__":
    # Example usage
    file_path = "data\\Vestas_RTP\\Documents\\User_guides\\User guide for the ready-to_x0002_protect (RtoP) system.pdf"

    guides_df = extract_userguide(file_path)