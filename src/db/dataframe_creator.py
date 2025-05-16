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
from src.utils.utils import log, match, SuppressStderr
from src.db.db_functions import get_documents_table, get_images_table
from src.ingestion.pdf_parser import create_documents_row
from src.ingestion.image_extractor import ocr_on_side_label_vestas_page
from src.ingestion.image_extractor import extract_image_data_from_page




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



def prepare_documents_df(pdf_files_path: str) -> pd.DataFrame:
    """
    Prepares a DataFrame of documents used for ingestion for the Documents table.
    The reason this function is separate from the create_documents_table function is that the VGA guide is parsed seperately from the other documents.
    Args:
        pdf_files_path (str): Path to the directory containing PDF files.
    """
    document_rows = []
    for idx, filename in enumerate(os.listdir(pdf_files_path)):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_files_path, filename)
            log(f"Document number: {idx}  : {file_path}", level=1)


            with SuppressStderr():
                with pdfplumber.open(file_path) as pdf:
                    plumber_text = pdf.pages[0].extract_text() or ""
                
                plumber_dict = create_documents_row(filename=filename, 
                                                    file_path = file_path, 
                                                    text = plumber_text)

                ocr_text = ocr_on_side_label_vestas_page(file_path = file_path)

                ocr_dict = create_documents_row(filename=filename, 
                                                file_path = file_path, 
                                                text = ocr_text)

                # Combine the two dictionaries, where we consider the plumber_dict as the main source of truth.
                for key, item in plumber_dict.items():
                    if item == None:
                        if ocr_dict[key] != None:
                            plumber_dict[key] = ocr_dict[key]
                plumber_dict["CONFIDENTIALITY"] = plumber_dict["CONFIDENTIALITY"].upper() if plumber_dict["CONFIDENTIALITY"] else None

                document_rows.append(plumber_dict)
            
    documents_df = pd.DataFrame(document_rows)
    return documents_df




if __name__ == "__main__":
    pass