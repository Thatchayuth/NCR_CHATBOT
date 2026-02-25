import pyodbc

# MSSQL Connection Settings
DB_SERVER = "NBK096"
DB_NAME = "PhoneDB"
DB_USER = "sa"
DB_PASSWORD = "Password123!"
DB_DRIVER = "{ODBC Driver 17 for SQL Server}"

def get_connection_string(database=None):
    """Get connection string for MSSQL"""
    db = database or DB_NAME
    return (
        f"DRIVER={DB_DRIVER};"
        f"SERVER={DB_SERVER};"
        f"DATABASE={db};"
        f"UID={DB_USER};"
        f"PWD={DB_PASSWORD};"
    )

def get_connection(database=None):
    """Get a pyodbc connection to MSSQL"""
    conn_str = get_connection_string(database)
    return pyodbc.connect(conn_str)

def get_master_connection():
    """Get connection to master database (for creating databases)"""
    return get_connection(database="master")
