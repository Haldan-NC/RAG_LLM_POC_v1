from langchain_community.document_loaders import PDFPlumberLoader
import pandas as pd

def extract_text_chunks(file_path: str, manual_id: int, chunk_size: int = 512, chunk_overlap: int = 128) -> pd.DataFrame:
    """
    Extracts text chunks from a PDF file, tracking the page numbers and creating a DataFrame.
    Args:
        file_path (str): Path to the PDF file.
        manual_id (int): Manual ID for the document.
        chunk_size (int): Size of each text chunk.
        chunk_overlap (int): Overlap between chunks.
    """
    loader = PDFPlumberLoader(file_path)
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



# Table names are either CHUNKS_LARGE or CHUNKS_SMALL
# Chunk size is either 7000 or 1024
# Chunk overlapp is 128 or 64




if __name__ == "__main__":
    pass