import mysql.connector
from mysql.connector import Error
import streamlit as st

def get_connection():
    """Membuat koneksi ke database MySQL."""
    try:
        conn = mysql.connector.connect(
           host=st.secrets["DB_HOST"],
            database=st.secrets["DB_DATABASE"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASSWORD"]
        )
        return conn
    except Error as e:
        print(f"Error saat menyambungkan ke database: {e}")
        return None
