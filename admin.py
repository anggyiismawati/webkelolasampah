import streamlit as st
import olahdata  # Mengimpor fungsi pengelolaan data dari olahdata.py
import olahinformasi  # Mengimpor fungsi pengelolaan informasi dari olahinformasi.py
import olahmodel  # Mengimpor fungsi pengelolaan model dari olahmodel.py
import logout  # Mengimpor fungsi logout dari logout.py

def admin_dashboard():
    """Halaman Admin yang hanya dapat diakses setelah login."""
    st.title(f"Selamat datang, {st.session_state.username}!")

    # Inisialisasi state untuk menu jika belum ada
    if "menu" not in st.session_state:
        st.session_state.menu = "Pengelolaan Data"  # Default menu

    # Sidebar dengan button navigasi dan ikon
    st.sidebar.title("Menu Admin")
    
    # Menggunakan markdown untuk menambahkan ikon dengan link
    if st.sidebar.button("Pengelolaan Data 🗃️"):
        st.session_state.menu = "Pengelolaan Data"
    if st.sidebar.button("Kelolaan Informasi 📑"):
        st.session_state.menu = "Pengelolaan Informasi"
    if st.sidebar.button("Hasil Model 🧠"):
        st.session_state.menu = "Hasil Model"
    if st.sidebar.button("Logout 🚪"):
        st.session_state.menu = "Logout"

    # Arahkan ke fungsi berdasarkan menu yang aktif di session state
    if st.session_state.menu == "Pengelolaan Data":
        olahdata.pengelolaan_data()
    elif st.session_state.menu == "Pengelolaan Informasi":
        olahinformasi.pengelolaan_informasi()
    elif st.session_state.menu == "Hasil Model":
        olahmodel.pengelolaan_model()
    elif st.session_state.menu == "Logout":
        st.session_state.clear()  # Menghapus seluruh session state
        st.rerun()  # Untuk memuat ulang aplikasi setelah logout
