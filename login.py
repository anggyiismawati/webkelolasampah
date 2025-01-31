import streamlit as st
from database import get_connection
from mysql.connector import Error

def verify_user(username, password):
    """Memverifikasi username dan password dari database."""
    conn = get_connection()
    if conn is None:
        return False
    
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM users WHERE username = %s AND password = %s"
        cursor.execute(query, (username, password))
        result = cursor.fetchone()
        conn.close()
        
        # Jika ditemukan, return True, jika tidak return False
        return result is not None
    except Error as err:  # Menggunakan Error dari mysql.connector
        st.error(f"Error saat melakukan query: {err}")
        conn.close()
        return False

def login():
    """Halaman login dengan Streamlit."""
    # Mengecek apakah user sudah login
    if 'logged_in' in st.session_state and st.session_state.logged_in:
        st.success(f"Selamat datang, {st.session_state.username}!")
        st.write("Anda sudah login. Klik tombol logout untuk keluar.")
        return
    
    # Styling CSS untuk mempercantik halaman login dengan warna cerah
    st.markdown("""
  <style>
    .login-container {
        background-color: #fffbf2;
        padding: 40px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .login-container h1 {
        text-align: center;
        color: #3366cc;
        font-size: 2em;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .login-container .stButton button {
        background-color: #3366cc;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 5px;
        padding: 12px 20px;
        width: 100%;
        cursor: pointer;
        transition: all 0.3s ease;
        border: none;
    }
    .login-container .stButton button:hover {
        background-color: #5b88e1;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
        padding: 12px;
        font-size: 18px;
        width: 100%;
        border: 1px solid #ccc;
        background-color: #f0f9ff;
    }
    .stTextInput>div>div>input:focus {
        border-color: #3366cc;
        box-shadow: 0 0 5px rgba(51, 102, 204, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Layout dan form login
    # st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.title("Login Halaman")
    
    # Input untuk username dan password dengan placeholder
    username = st.text_input("Username :", placeholder="Masukkan username Anda")
    password = st.text_input("Password :", type='password', placeholder="Masukkan password Anda")
    
    # Tombol login
    if st.button("Login", key="login_button"):
        if username and password:
            if verify_user(username, password):
                st.success("Login berhasil!")
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Username atau password salah!")
        else:
            st.error("Username dan password harus diisi.")
    st.markdown('</div>', unsafe_allow_html=True)
