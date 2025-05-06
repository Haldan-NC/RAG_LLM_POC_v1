from openai import OpenAI
import keyring
import re
import json
import sys
sys.path.append("..\\.")  
sys.path.append("..\\..\\.") 

from src.utils.utils import get_config


def get_openai_api_key() -> str:
    """
    Retrieves OpenAI API key from Windows Credential Manager.
    """
    cfg = get_config()
    open_ai_api_key = keyring.get_password(cfg['windows_credential_manager']['openai']['api_key'], 'api_key')
    
    if open_ai_api_key is None:
        raise ValueError("OpenAI API key not found in Windows Credential Manager.")
    
    return open_ai_api_key


def get_openai_client() -> OpenAI:
    """
    Returns an OpenAI client instance.
    """
    open_ai_api_key = get_openai_api_key()
    client = OpenAI(api_key=open_ai_api_key)
    return client


def generate_promt_for_openai_api(instructions, input_text) -> str:
    client = get_openai_client()
    response = client.responses.create(
        model="gpt-4o",
        instructions= instructions,
        input=input_text
    )

    return response


def extract_json_from_open_ai_llm_output(llm_output_text: str) -> dict:
    """
    Made and tested for the OpenAI LLM output.
    This function extracts JSON from the LLM output text.
    """
    try:
        # Look for a code block marked with ```json ... ```
        match = re.search(r"```json\s*(\{.*?\})\s*```", llm_output_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON code block found in the text.")

        raw_json = match.group(1)

        # Optional: Remove trailing commas which are invalid in JSON
        cleaned_json = re.sub(r",\s*([\]}])", r"\1", raw_json)

        parsed = json.loads(cleaned_json)
        return parsed

    except Exception as e:
        print("Failed to extract JSON:", e)
        return {}
