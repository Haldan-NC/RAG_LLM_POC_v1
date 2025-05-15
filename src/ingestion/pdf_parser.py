import pdfplumber
import pandas as pd
from langchain_community.document_loaders import PDFPlumberLoader
import sys
sys.path.append(".")  
sys.path.append("..\\.")  
sys.path.append("..\\..\\.") 
import os
import re

from src.ingestion.image_extractor import extract_images_from_pdf, generate_image_table
from src.ingestion.image_extractor import ocr_on_side_label_vestas_page
from src.utils.utils import log, match, SuppressStderr


def extract_text_chunks(file_path: str, manual_id: int, chunk_size: int = 512, chunk_overlap: int = 128) -> pd.DataFrame:
    """
    Extracts text chunks from a PDF file, tracking the page numbers and creating a DataFrame.
    Args:
        file_path (str): Path to the PDF file.
        manual_id (int): Manual ID for the document.
        chunk_size (int): Size of each text chunk.
        chunk_overlap (int): Overlap between chunks.
    """
    loader = PDFPlumberLoader(file_path)  # Change this to the other pdfplumber package in the future
    docs = loader.load()

    # Step 1: Combine all text across pages with page tracking
    all_text = ""
    page_map = []  # (char_index, page_number)

    for doc_page in docs:
        text = doc_page.page_content.strip().replace('\n', ' ')
        start_idx = len(all_text)
        all_text += text + " "  # Add space to separate pages
        end_idx = len(all_text)
        page_map.append((start_idx, end_idx, doc_page.metadata['page']))

    # Step 2: Create chunks with overlap, spanning across pages
    chunks = []
    chunk_order = []
    page_start_list = []
    page_end_list = []

    idx = 0
    chunk_idx = 0

    while idx < len(all_text):
        chunk = all_text[idx:idx + chunk_size]

        # Determine pages involved in this chunk
        chunk_start = idx
        chunk_end = idx + len(chunk)

        pages_in_chunk = [
            page_num
            for start, end, page_num in page_map
            if not (end <= chunk_start or start >= chunk_end)  # overlap condition
        ]

        page_start = min(pages_in_chunk) if pages_in_chunk else None
        page_end = max(pages_in_chunk) if pages_in_chunk else None

        chunks.append(chunk)
        page_start_list.append(page_start)
        page_end_list.append(page_end)
        chunk_order.append(chunk_idx)

        chunk_idx += 1
        idx += chunk_size - chunk_overlap

    # Step 3: Create DataFrame
    rows = [{
        'DOCUMENT_ID': manual_id,
        'PAGE_START_NUMBER': start,
        'PAGE_END_NUMBER': end,
        'CHUNK_TEXT': chunk,
        'CHUNK_ORDER': order
    } for chunk, start, end, order in zip(chunks, page_start_list, page_end_list, chunk_order)]

    df = pd.DataFrame(rows, columns=["DOCUMENT_ID", "PAGE_START_NUMBER", "PAGE_END_NUMBER", "CHUNK_TEXT", "CHUNK_ORDER"])
    return df


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


def extract_metadata_regex_patterns() -> dict:
    """
    Returns a dictionary of regex patterns for extracting metadata fields from text.
    """
    patterns = {
        "DMS_NO": re.compile(r"\b\d{4}-\d{4}\b"),
        "VERSION": re.compile(r"\bVER[:\s]+(\d{2})\b", re.IGNORECASE),
        "VERSION_DATE": re.compile(r"Date[:\s]*([\d]{4}-[\d]{2}-[\d]{2})", re.IGNORECASE),
        "EXPORTED_DATE": re.compile(r"Exported from DMS.*?(\d{4}-\d{2}-\d{2})", re.IGNORECASE),
        "DOC_TYPE": re.compile(r"\bT(?:\d{2}|[oO]\d)\b"),
        "CONFIDENTIALITY": re.compile(r"\b(Confidential|Restricted)\b(?![a-zA-Z])", re.IGNORECASE),
        "APPROVED": re.compile(r"\bApproved\b", re.IGNORECASE)
    }

    return patterns


def create_documents_row(filename: str, file_path: str, text: str) -> dict:
    """
    Creates a row for the documents table.
    Args:
        filename (str): Name of the PDF file.
        file_path (str): Path to the PDF file.
        text (int): text of the front page of the PDF file.
    Returns:
        dict: Row for the documents table.
    """
    patterns = extract_metadata_regex_patterns()

    doc_type_match = patterns["DOC_TYPE"].search(text)
    doc_type = doc_type_match.group(0).replace("O", "0") if doc_type_match else None
    file_size = os.path.getsize(file_path)

    return {
        "DOCUMENT_NAME": filename,
        "FILE_PATH": file_path,
        "DMS_NO": match(pattern = patterns["DMS_NO"], text = text, group_index = 0),
        "VERSION": match(pattern = patterns["VERSION"], text = text, group_index = 0),
        "VERSION_DATE": match(pattern = patterns["VERSION_DATE"], text = text, group_index = 0),
        "EXPORTED_DATE": match(pattern = patterns["EXPORTED_DATE"], text = text, group_index = 0),
        "DOC_TYPE": doc_type,
        "CONFIDENTIALITY": match(pattern = patterns["CONFIDENTIALITY"], text = text, group_index = 0),
        "APPROVED": "Y" if patterns["APPROVED"].search(text) else None,
        "FILE_SIZE": file_size
    }




if __name__ == "__main__":
    pass