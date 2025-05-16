import pdfplumber
import pandas as pd
from langchain_community.document_loaders import PDFPlumberLoader
import sys
sys.path.append(".")  
sys.path.append("..\\.")  
sys.path.append("..\\..\\.") 
import os
import re
from tqdm import tqdm

from src.utils.utils import log
from src.ingestion.image_extractor import extract_image_data_from_page
from src.utils.utils import log, match, SuppressStderr
from src.db.db_functions import get_documents_table, get_images_table




def prepare_images_df(image_dest: str) -> pd.DataFrame:
    """
    Creates a DataFrame of images used for ingestion for the Images table.
    """
    documents_df = get_documents_table()
    image_df = get_images_table()
    image_data = []    
    
    for document_index, document_path in tqdm(enumerate(documents_df["FILE_PATH"]), total = len(documents_df), desc = f"Extracting images from {len(documents_df)} PDFs"):
        with SuppressStderr():
            row = documents_df.iloc[document_index]
            document_id = row["DOCUMENT_ID"]
            document_name = row["DOCUMENT_NAME"]
            if document_id in image_df["DOCUMENT_ID"].values:
                # If document allready processed, continue
                continue
            with pdfplumber.open(document_path) as pdf:
                for page_num, page_object in enumerate(pdf.pages, 1):
                    image_data += extract_image_data_from_page(page_object, page_num, document_name, document_id, image_dest)

    images_df = pd.DataFrame(image_data)
    return images_df





if __name__ == "__main__":
    pass