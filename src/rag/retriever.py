import pandas as pd
from typing import Optional, Dict, List
from dataclasses import dataclass
import sys
sys.path.append("..\\.")  
sys.path.append("..\\..\\.") 
sys.path.append("..\\..\\..\\.") 
from src.db.db_functions import get_cursor
from src.utils.utils import log
from src.llm_functions.cortex_llm_functions import vector_embedding_cosine_similarity_between_texts


@dataclass
class DocumentInfo:
    """
    Data class to hold document information with type safety.
    Includes a similarity score for ranking multiple matches.
    """
    document_id: int
    document_name: str
    doc_version: str
    file_path: str
    similarity_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "DOCUMENT_ID": self.document_id,
            "DOCUMENT_NAME": self.document_name,
            "DOC_VERSION": self.doc_version,
            "FILE_PATH": self.file_path,
            "SIMILARITY_SCORE": self.similarity_score
        }


def find_documents_by_machine_name(
    machine_name: str, 
    similarity_threshold: float = 0.7
) -> List[DocumentInfo]:
    """
    Find relevant documents for a given machine name using multiple matching strategies.
    """
    if not machine_name or not isinstance(machine_name, str):
        raise ValueError("Machine name must be a non-empty string")
    
    log(f"Searching for documents matching machine: {machine_name}", level=1)
    
    try:
        conn, cursor = get_cursor()
        matching_docs: List[DocumentInfo] = []
        seen_document_ids = set()
        
        # Get all documents
        cursor.execute("""
            SELECT DOCUMENT_ID, DOCUMENT_NAME, DOC_VERSION, FILE_PATH
            FROM DOCUMENTS
            WHERE 1=1;
        """)
        documents_df = cursor.fetch_pandas_all()
        
        if documents_df.empty:
            log("No documents found in database", level=1)
            return []

        # Strategy 1: Exact match
        exact_matches = documents_df[
            documents_df['DOCUMENT_NAME'].str.contains(
                f"\\b{machine_name}\\b", 
                case=False, 
                regex=True
            )
        ]
        
        for _, doc in exact_matches.iterrows():
            if doc['DOCUMENT_ID'] not in seen_document_ids:
                seen_document_ids.add(doc['DOCUMENT_ID'])
                matching_docs.append(
                    DocumentInfo(
                        document_id=doc['DOCUMENT_ID'],
                        document_name=doc['DOCUMENT_NAME'],
                        doc_version=doc['DOC_VERSION'],
                        file_path=doc['FILE_PATH'],
                        similarity_score=1.0
                    )
                )
        
        # Strategy 2: Check VGA Guides table for machine model references
        if not matching_docs:  # Only check VGA guides if no exact matches found
            cursor.execute("""
                SELECT DISTINCT d.DOCUMENT_ID, d.DOCUMENT_NAME, d.DOC_VERSION, d.FILE_PATH, g.TURBINE_MODELS
                FROM DOCUMENTS d
                JOIN VGA_GUIDES g ON d.DOCUMENT_ID = g.DOCUMENT_ID
                WHERE g.TURBINE_MODELS LIKE %s;
            """, (f"%{machine_name}%",))
            
            vga_matches = cursor.fetch_pandas_all()
            for _, doc in vga_matches.iterrows():
                if doc['DOCUMENT_ID'] not in seen_document_ids:
                    seen_document_ids.add(doc['DOCUMENT_ID'])
                    matching_docs.append(
                        DocumentInfo(
                            document_id=doc['DOCUMENT_ID'],
                            document_name=doc['DOCUMENT_NAME'],
                            doc_version=doc['DOC_VERSION'],
                            file_path=doc['FILE_PATH'],
                            similarity_score=0.9
                        )
                    )
        
        # Strategy 3: Semantic similarity (only if no other matches found)
        if not matching_docs:
            log("No exact matches found, trying semantic similarity", level=2)
            for _, doc in documents_df.iterrows():
                if doc['DOCUMENT_ID'] not in seen_document_ids:
                    similarity = vector_embedding_cosine_similarity_between_texts(
                        machine_name,
                        doc['DOCUMENT_NAME']
                    )
                    
                    if similarity >= similarity_threshold:
                        seen_document_ids.add(doc['DOCUMENT_ID'])
                        matching_docs.append(
                            DocumentInfo(
                                document_id=doc['DOCUMENT_ID'],
                                document_name=doc['DOCUMENT_NAME'],
                                doc_version=doc['DOC_VERSION'],
                                file_path=doc['FILE_PATH'],
                                similarity_score=similarity
                            )
                        )
        
        # Sort by similarity score
        matching_docs.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Single log statement for results
        log(f"Found {len(matching_docs)} matching documents", level=1)
        for doc in matching_docs:
            log(f"Match: {doc.document_name} (score: {doc.similarity_score:.2f})", level=2)
        
        return matching_docs
    
    except Exception as e:
        log(f"Error finding documents: {str(e)}", level=0)
        raise RuntimeError(f"Failed to search documents: {str(e)}")
    
    finally:
        conn.close()


def get_best_document_for_machine(machine_name: str) -> Optional[Dict]:
    """
    Get the most relevant document for a given machine name.
    This is a wrapper function that returns the best match or None.
    
    Args:
        machine_name (str): The machine name to search for
        
    Returns:
        Optional[Dict]: Best matching document info or None if no matches found
    """
    try:
        matches = find_documents_by_machine_name(machine_name)
        if matches:
            return matches[0].to_dict()
        return None
        
    except Exception as e:
        log(f"Error getting best document: {str(e)}", level=0)
        return None


def narrow_down_relevant_chunks(task_chunk_df: pd.DataFrame, document_info: dict) -> pd.DataFrame:
    """Filter chunks to only include those from the specified document."""
    return task_chunk_df[task_chunk_df['DOCUMENT_ID'] == document_info['DOCUMENT_ID']]


def pick_image_based_of_descriptions(image_candidates: pd.DataFrame, step_text: str) -> str:
    image_options_text = ""
    for _, image_row in image_candidates.iterrows():
        image_id = image_row["IMAGE_ID"]
        image_path = image_row["IMAGE_PATH"]
        description = image_row["DESCRIPTION"]
        image_options_text += f"- Image ID: {image_id}, Path: {image_path}, Description: {description}\n"

    instructions = f"""
    You are tasked with modifying the task in a step by step guide. You will append the most relevant image reference to the step,
    by selecting the most relevant image for the following step in a guide:
    "{step_text}"
    """

    reference_text = f"""
    ### Image Options:
    {image_options_text}
    """

    response = generate_promt_for_openai_api(instructions, input_text)
    return response.output_text