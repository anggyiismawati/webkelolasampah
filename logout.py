import streamlit as st

def logout():
    """Fungsi untuk logout pengguna."""
    # Reset session state untuk logout
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.page = "Login"  # Arahkan ke halaman login
    st.success("Logout berhasil! Silakan login kembali.")
    st.rerun()  # Reload tampilan
