## This is a guide on how to set up the project locally.

## 1. Clone the repository

## 2. Install requirements.txt with pip

## 3. Setup the credentials for Snowflake and OpenAI in the credential manager

## 4. Create a directory in "data" called "Vestas_RTP/Documents" and place the PDF files in there. Do not commit the PDF files to the repository.

## 5. Run the following python script in the root directory of the project to set up the database and tables:

```bash
cd "RAG_LLM_POC_v1" 
python setup/vestas_database_setup.py
```
