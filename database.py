import mysql.connector
from mysql.connector import Error

def get_connection():
    """Membuat koneksi ke database MySQL."""
    try:
        conn = mysql.connector.connect(
            host='localhost',        # Ganti dengan host MySQL Anda
            database='kelola_sampah',  # Ganti dengan nama database Anda
            user='root',             # Ganti dengan username MySQL Anda
            password=''   # Ganti dengan password MySQL Anda
        )
        return conn
    except Error as e:
        print(f"Error saat menyambungkan ke database: {e}")
        return None
