## Vestas Virtual Tech – PoC #1: RAG-LMM

This proof-of-concept establishes a structured foundation for ongoing development, providing a modular codebase that invites focused, code-style contributions to individual components.

---

### 1. Installation

1. **Clone the repository**  
   ```bash
   git clone <repo-url>
   cd RAG_LLM_POC_v1
   ```
2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Do not alter** `connection_config.yaml` _(see section 3.5)_.  
4. **Setup credentials for Snowflake and OpenAI**  
   The codebase uses Python’s `keyring` library to interact with Windows Credential Manager. Follow the examples below:

   **Example 1: Snowflake**  
   - In **Credential Manager → Windows Credentials → Add a generic credential**  
     - Internet or network address: `NC_Snowflake_Trial_Account_Name`  
     - User name: `account_identifier`  
     - Password: `<your_snowflake_account_identifier>`

   **Example 2: OpenAI**  
   - In **Credential Manager → Windows Credentials → Add a generic credential**  
     - Internet or network address: `OpenAI_API_Key`  
     - User name: `api_key`  
     - Password: `<your_openai_api_key>`

5. **Initialize the database and tables**  
   > _Must be run from the project root:_
   ```bash
   cd RAG_LLM_POC_v1
   python setup/washing_machine_database_setup.py
   ```

---

### 2. Example Usage

1. **Create database and tables**  
   ```bash
   cd RAG_LLM_POC_v1
   python setup/washing_machine_database_setup.py
   ```
2. **Run the RAG pipeline**  
   ```bash
   cd RAG_LLM_POC_v1
   python src/rag/app.py
   ```

---

### 3. Project Structure

```text
RAG_LLM_POC_v1/
├── docs/
├── setup/
│   ├── README.txt
│   └── washing_machine_database_setup.py
├── data/
│   ├── [PDF files]
│   └── [Extracted images]
├── src/
│   ├── db/
│   │   └── db_functions.py
│   ├── ingestion/
│   │   ├── image_extractor.py
│   │   ├── pdf_parser.py
│   │   └── llm_functions/
│   │       └── [LLM-specific modules]
│   ├── rag/
│   │   ├── app.py
│   │   ├── generator.py
│   │   └── retriever.py
│   └── utils/
│       └── [Utility modules]
├── tests/
│   └── readme.txt
└── config/
    └── connection_config.yaml
```

- **setup/**: Database-initialization scripts (currently for a washing-machine example; replace with Vestas data).  
- **data/**: Source PDFs and their extracted images.  
- **src/db/**: Functions for Snowflake integration.  
- **src/ingestion/**: ETL and ingestion logic, including `llm_functions/`.  
- **src/rag/**: Retrieval-Augmented Generation pipeline entry point.  
- **src/utils/**: Shared utility code.  
- **tests/**: Test pipelines for performance and benchmarking.  
- **config/**: Connection settings (e.g. `connection_config.yaml`), replaceable with Azure Key Vault or environment variables in the future.
