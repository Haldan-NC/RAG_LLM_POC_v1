## Vestas Virtual Tech – PoC #1: RAG-LMM

This proof-of-concept establishes a structured foundation for ongoing development, providing a modular codebase that invites focused, code-style contributions to individual components.

#### The readme is structured as follows:
#### 1. Installation
#### 2. Example Usage and Database Setup
#### 3. Project Structure


---

### 1. Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Haldan-NC/RAG_LLM_POC_v1
   cd RAG_LLM_POC_v1
   ```
2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Do not alter** `connection_config.yaml` _(see section 4)_.  
4. **Setup credentials for Snowflake and OpenAI**  
   The codebase uses Python’s `keyring` library to interact with Windows Credential Manager. Follow the examples below:

   **Snowflake Account Identifier:**  
   - In **Credential Manager → Windows Credentials → Add a generic credential**  
     - Internet or network address: `NC_Snowflake_Trial_Account_Name`  
     - User name: `account_identifier`  
     - Password: `<your snowflake account identifier: <Orginization Name>-<Account Name>>`

   **Snowflake User Name:**  
   - In **Credential Manager → Windows Credentials → Add a generic credential**  
     - Internet or network address: `NC_Snowflake_Trial_User_Name`  
     - User name: `user_name`  
     - Password: `<your snowflake user name>`

   **Snowflake Account Identifier:**  
   - In **Credential Manager → Windows Credentials → Add a generic credential**  
     - Internet or network address: `NC_Snowflake_Trial_User_Password`  
     - User name: `password`  
     - Password: `<your snowflake password>`

   **OpenAI:**  
   - In **Credential Manager → Windows Credentials → Add a generic credential**  
     - Internet or network address: `OpenAI_API_Key`  
     - User name: `api_key`  
     - Password: `<your_openai_api_key>`


---

### 2. Example Usage and Database Setup

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

3. **Run model evaluation**  
   ```bash
   cd RAG_LLM_POC_v1
   python tests/T.B.D
   ```

---

### 3. Project Structure

```text
RAG_LLM_POC_v1/
├── config/
│   └── connection_config.yaml
├── docs/
├── setup/
│   └── washing_machine_database_setup.py
├── data/
│   └── Washing_Machine_Data
│       ├── Documents [PDF files]
│       └── Images [Extracted images]
├── src/
│   ├── db/
│   │   └── db_functions.py
│   ├── ingestion/
│   │   ├── image_extractor.py
│   │   ├── pdf_parser.py
│   │   └── llm_functions/
│   │       ├── cortex_llm_functions.py
│   │       └── openai_llm_functions.py
│   ├── rag/
│   │   ├── app.py
│   │   ├── generator.py
│   │   └── retriever.py
│   └── utils/
│       ├── cortex_utils.py
│       ├── openai_utils.py
│       └── utils.py
└── tests/
    └── readme.txt
```

- **setup/**: Database-initialization scripts (currently for a washing-machine example; replace with Vestas data).  
- **data/**: Source PDFs and their extracted images.  
- **src/db/**: Functions for Snowflake integration.  
- **src/ingestion/**: ETL and ingestion logic, including `llm_functions/`.  
- **src/rag/**: Retrieval-Augmented Generation pipeline entry point.  
- **src/utils/**: Shared utility code.  
- **tests/**: Test pipelines for performance and benchmarking.  
- **config/**: Connection files for windows credential manager (e.g. `connection_config.yaml`), replaceable with Azure Key Vault or environment variables in the future.
