import streamlit as st
import admin  # Mengimpor halaman admin dari file admin.py
from beranda import beranda  # Mengimpor fungsi beranda dari beranda.py
from login import login  # Mengimpor fungsi login dari login.py
from informasi import informasi  # Mengimpor fungsi informasi dari informasi.py
from hasil_model import hasil_model  # Mengimpor fungsi hasil_model dari hasil_model.py

def main():
    """Main function untuk memeriksa apakah pengguna sudah login atau belum."""
    # Set page config untuk mengubah judul dan ikon halaman
    st.set_page_config(
        page_title="Website Kelola Sampah",  # Judul tab browser
        page_icon="♻️",  # Ikon tab browser (gunakan emoji atau path gambar)
    )
    
    # Sidebar dengan button navigasi dan ikon
    st.sidebar.title("Menu Utama")
    
    # Menambahkan sidebar dengan tombol dan ikon
    if st.sidebar.button("🏠 Beranda", key="home_button"):
        st.session_state.page = "Beranda"
    
    if st.sidebar.button("ℹ️ Informasi", key="info_button"):
        st.session_state.page = "Informasi"
    
    if st.sidebar.button("📊 Hasil Model", key="model_button"):
        st.session_state.page = "Hasil Model"
        
    if st.sidebar.button("🔑 Login", key="login_button_sidebar"):
        st.session_state.page = "Login"
    
    # Memeriksa halaman yang dipilih
    if 'page' not in st.session_state:
        st.session_state.page = "Beranda"

    if st.session_state.page == "Login":
        if 'logged_in' not in st.session_state or not st.session_state.logged_in:
            login()  # Memanggil fungsi login dari login.py
        else:
            st.session_state.page = "Admin"
            st.rerun()
    elif st.session_state.page == "Beranda":
        beranda()  # Memanggil fungsi beranda dari beranda.py
    elif st.session_state.page == "Informasi":
        informasi()  # Memanggil fungsi informasi dari informasi.py
    elif st.session_state.page == "Hasil Model":
        hasil_model()  # Memanggil fungsi hasil_model dari hasil_model.py
    elif st.session_state.page == "Admin":
        admin.admin_dashboard()  # Memanggil fungsi admin_dashboard dari admin.py

# Menjalankan aplikasi
main()
