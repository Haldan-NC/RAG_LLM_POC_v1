




def extract_TOC_cortex(text: str, model : str = 'llama3.1-70b') -> str:
    """
    Extracts the table of contents from a given text using Cortex LLM.
    The function sends a prompt to the Cortex LLM and retrieves the response.

    NOTE: This function is not tested yet in this environment. It is a scaffolding code that worked in a notebook.
    """

    raise NotImplementedError("Cortex Complete in this environemtn is not tested yet. This is purely scaffolding code that worked in a notebook.")
    prompt = (
    f"""
    I will provide a long string of text that most likely contains a table of contents, 
    although it may also include additional body text from a document. Your task is to carefully 
    extract only the table of contents and structure it as a JSON object in the following 
    format:
    {
      "Section": "<section name>",
      "Section Number": "<section name>",
      "Page": <page number>,
      "Sub Sections" : [{
        "Section": "<section name>",
        "Section Number": "<section name>",
        "Page": <page number>,
        "Sub Sections" : []}
      ],
    }    

    Guideines:
    - All keys in the json object must be either "Section", "Section Number", "Page", "Sub Sections".
    - "Section Number" must be represented as an integer or float - E.G: 1, 2, 5.3, 1,4, etc.
    - Ignore any text that is not part of the table of contents.
    - Ensure that sub-sections are nested appropriately under their parent section.
    - Page numbers should be extracted as integers, if possible.
    - Be tolerant of inconsistencies in formatting, spacing, or punctuation (e.g. dashes, colons, ellipses).
    - Do not include duplicate or repeated sections.
    - You should only consider items which are part of the table of contents, nothing before, nothing after.
    - "Section" must consist of words
    - "Section Number" must be represented as an integer or float - E.G: 1, 2, 5.3, 1,4, etc.
    - You must include a top level key value pair called "Section":"Table of contents".

    
    Text:
    {text}
    """)


    conn, cursor = get_cursor()
    start_time = time.time()
    result = cursor.execute(f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE('{model}', $$ {prompt} $$)
    """)
    print(f"Runtime in seconds: {time.time() - start_time:.4f}")

    return cursor.fetch_pandas_all().iloc[0,0]
