import streamlit as st
from olahmodel import pengelolaan_model

def hasil_model():
    """Fungsi untuk halaman Hasil Model."""

    # Menambahkan CSS kustom untuk mengubah font dan gaya
    st.markdown("""
    <style>
        body {
            font-family: 'Helvetica Neue', sans-serif;
            background-color: #f4f6f9;
            color: #2c3e50;
        }
        h1 {
            font-family: 'Georgia', serif;
            color: #2c3e50;
            font-size: 36px;
            font-weight: bold;
        }
        h2 {
            font-family: 'Arial', sans-serif;
            color: #8e44ad;
            font-size: 24px;
        }
        p {
            font-family: 'Verdana', sans-serif;
            color: #34495e;
            font-size: 16px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Memanggil fungsi pengelolaan_model dari olahmodel.py
    try:
        pengelolaan_model()  # Fungsi ini sudah mencakup seluruh proses dan visualisasi model
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses model: {e}")

# Menjalankan aplikasi Streamlit dengan pemanggilan fungsi hasil_model()
if __name__ == "__main__":
    hasil_model()
