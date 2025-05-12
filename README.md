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

3. **Create a directories in `data` called `Vestas_RTP/Documents/Documents` and `Vestas_RTP/Documents/VGA_guides`**
   - Place "No communication Rtop - V105 V112 V117 V126 V136 3,3-4,2MW MK3.pdf" in the VGA_guides folder.
   - Place "0078-6200_V07 - 0078-6200_4MW Mk3E Setting and Adjustment of Relays.pdf" in the Documents folder.
   - **Note:** The PDF files are not included in the repository.
   - **Note:** The pipeline is currently only designed to work with the VGA guide! Using other documents will break the pipeline!
   - **Do not commit the PDF files to the repository.**  
   - The directory structure should look like this:  
     ```text
     RAG_LLM_POC_v1/
     └── data/
         └── Vestas_RTP/
             └── Documents/
                 ├── Documents/
                 |   └── 0078-6200_V07 - 0078-6200_4MW Mk3E Setting and Adjustment of Relays.pdf
                 └── VGA_guides/
                    └── No communication Rtop - V105 V112 V117 V126 V136 3,3-4,2MW MK3.pdf

     ```

4. **Do not alter** `connection_config.yaml` _(see section 5)_.  

5. **Setup credentials for Snowflake and OpenAI**  
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
   python setup/vestas_database_setup.py
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

4. **Change the verbosity_level in `config/log_config.yaml` for detailed logging.**  


---

### 3. Project Structure

```text
RAG_LLM_POC_v1/
├── config/
│   ├── connection_config.yaml
│   └── log_config.yaml
├── setup/
│   ├── washing_machine_database_setup.py
│   └── vestas_database_setup.py
├── data/
│   ├── Vestas_RTP/
│   │   ├── Documents/
│   │   │   └── Documents/
│   │   │       └── 0078-6200_V07 - 0078-6200_4MW Mk3E Setting and Adjustment of Relays.pdf
│   │   │   └── VGA_guides/
│   │   │       └── No communication Rtop - V105 V112 V117 V126 V136 3,3-4,2MW MK3.pdf
│   └── Washing_Machine_Data/
│       ├── Documents [PDF files]/
│       └── Images [Extracted images]/
├── src/
│   ├── db/
│   │   └── db_functions.py
│   ├── ingestion/
│   │   ├── image_extractor.py
│   │   ├── pdf_parser.py
│   │   ├── vga_pdf_parser.py
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

- **setup/**: Database-initialization scripts. 
- **data/**: Source PDFs and their extracted images.  
- **src/db/**: Functions for Snowflake integration.  
- **src/ingestion/**: ETL and ingestion logic, including `llm_functions/`.  
- **src/rag/**: Retrieval-Augmented Generation pipeline entry point.  
- **src/utils/**: Shared utility code.  
- **tests/**: Test pipelines for performance and benchmarking.  
- **config/**: Connection files for windows credential manager (e.g. `connection_config.yaml`), replaceable with Azure Key Vault or environment variables in the future.
