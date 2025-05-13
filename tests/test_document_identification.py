import os
import sys
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import the modules
from src.rag.retriever import find_documents_by_machine_name, get_best_document_for_machine
from src.utils.utils import log
from src.llm_functions.cortex_llm_functions import vector_embedding_cosine_similarity_between_texts

def test_document_identification():
    test_cases = [
        "V112",
        "V126",
        "V105",
        "NonexistentModel",
        ""  # Should handle empty string
    ]

    log("=== Testing New Implementation ===", level=0)
    for machine in test_cases:
        try:
            log(f"\nTesting machine: {machine}", level=1)
            
            # Test find_documents_by_machine_name
            log("Testing find_documents_by_machine_name:", level=1)
            matches = find_documents_by_machine_name(machine)
            
            # Test get_best_document_for_machine
            log("\nTesting get_best_document_for_machine:", level=1)
            result = get_best_document_for_machine(machine)
            if result:
                log(f"Best match:", level=1)
                log(f"- Document Name: {result['DOCUMENT_NAME']}", level=1)
                log(f"- Document ID: {result['DOCUMENT_ID']}", level=1)
                log(f"- Similarity Score: {result['SIMILARITY_SCORE']:.2f}", level=1)
                log(f"- File Path: {result['FILE_PATH']}", level=1)
            else:
                log("No matching document found", level=1)
        except Exception as e:
            log(f"Error during test: {str(e)}", level=0)

if __name__ == "__main__":
    test_document_identification()