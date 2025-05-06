
## Vestas Virtual Tech - PoC #1 RAG-LMM

This proof-of-concept establishes a structured foundation for ongoing development, providing a modular codebase that invites focused, code-style contributions to individual components.


1. Installation process is as follows:
    1.1 Clone the repository
    1.2 Install requirements.txt with pip
    1.3 Do not alter connection_config.yaml -> See step 3.4
    1.4 Setup the credentials for Snowflake and OpenAI. The codebase uses keyring, which interacts with the credential manager in Windows 
        Follow the specific example format (see below):
        - Example 1:
        1.4.1 Add a generic credential 
            - Internet or Network address: NC_Snowflake_Trial_Account_Name
            - User name: account_identifier
            - Password: <add your snowflake account_identifier here>
        - Example 2:
        1.4.2 Add a generic credential 
            - Internet or Network address: OpenAI_API_Key
            - User name: api_key
            - Password: <add your Open AI API key here>
    1.5 To set up the database and tables, run the following python script in the root directory of the project:
        - Note: The codebase will not work unless it is run in the root directory of the project.
        ```bash
        cd "RAG_LLM_POC_v1" 
        python setup/washing_machine_database_setup.py
        ```

2. Example usage:
    2.1 To run the database creation pipeline, run the following python script in the root directory of the project:
        ```bash
        cd "RAG_LLM_POC_v1" 
        python setup/washing_machine_database_setup.py
        ```
    
    2.2 To run the RAG pipeline, run the following python script in the root directory of the project:
        ```bash
        cd "RAG_LLM_POC_v1" 
        python src/rag/app.py
        ```

3. The project is structured as follows:
    3.1 setup: Contains the database setup script for washing machines, intended to be replaced with vestas data.
    3.2 data: Contains the data, specifically the PDF files and the generated images from the PDF files.
    3.3 src: Contains the source code for the project.
        3.3.1 db: Everything related to interacting with the Snowflake database. 
        3.3.2 ingestion: Everything related to ETL and data ingestion.
            3.3.2.1 llm_functions: Specific functions using the LLM API's.
        3.3.3 rag: Everything related to the RAG pipeline.
        3.3.4 utils: Utility functions for the project.
    3.4 tests: Contains the test pipelines to assess and benchmark performance of new implementations.
    3.5 config: Contains the configuration strings to connect to windows credential manager using keyring. (Can be replaced with Azure Key Vault or Environment Variables in the future).


    
