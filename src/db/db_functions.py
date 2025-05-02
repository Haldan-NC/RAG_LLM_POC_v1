import sys
sys.path.append("..\\.")  
sys.path.append("..\\..\\.") 
import os
import yaml
import keyring
import snowflake.connector as sf_connector
from snowflake.connector.pandas_tools import write_pandas
from src.utils.utils import get_config


def get_connection(account: str, user: str, password: str, database: str, schema: str):
    """
    Returns a Snowflake connection and cursor. Used by setup and RAG client.
    """
    conn = sf_connector.connect(account=account,
                                user=user, 
                                password=password, 
                                database=database,
                                schema=schema,
                                warehouse='COMPUTE_WH',
                                role='ACCOUNTADMIN')
    cursor = conn.cursor()
    return conn, cursor


def get_cursor():
    """
    Returns Snowflake cursor (for RAG lookup). Called at start of RAG pipeline.

    Database and schema are hardcoded for now, but should be added to the config file.
    """
    cfg = get_config()

    # These credentials need to be polished up such that they fit with the actualt crednetial manager
    acct = keyring.get_password(cfg['windows_credential_manager']['snowflake']['account_identifier'], 'account_identifier')
    user = keyring.get_password(cfg['windows_credential_manager']['snowflake']['user_name'], 'user_name')
    pwd  = keyring.get_password(cfg['windows_credential_manager']['snowflake']['password'], 'password')
    database = cfg['snowflake']['database']
    schema = cfg['snowflake']['schema']

    conn, cursor = get_connection(acct, user, pwd, database, schema)
    return conn, cursor



if __name__ == "__main__":

    # The cursor works now.
    test = get_cursor()