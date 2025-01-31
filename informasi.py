import streamlit as st
from database import get_connection  # Fungsi untuk koneksi ke database

def informasi():

    """Fungsi untuk halaman utama yang menampilkan konten Informasi."""
    st.title("Informasi Tersedia")

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

    # Membuat koneksi ke database
    conn = get_connection()
    if conn is None:
        st.error("Koneksi ke database gagal!")
        return

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT judul, deskripsi, tanggal, gambar FROM konten ORDER BY tanggal DESC")
        konten_data = cursor.fetchall()

        if konten_data:
            for konten in konten_data:
                judul, deskripsi, tanggal, gambar_path = konten

                # Menampilkan judul dengan format yang lebih menarik
                st.markdown(f"### **{judul}**")
                st.markdown(f"📅 **Tanggal**: {tanggal}")

                # Menampilkan deskripsi dengan pemformatan yang lebih jelas
                st.markdown(f"**Deskripsi**: {deskripsi}")

                if gambar_path:
                    # Membuat kolom untuk menempatkan gambar di tengah
                    col1, col2, col3 = st.columns([1, 4, 1])  # Menambahkan kolom di kiri, tengah, dan kanan
                    with col2:  # Menempatkan gambar di kolom tengah
                        st.image(gambar_path, caption=f"Gambar terkait: {judul}", width=400)

                # Memberikan pemisah antar konten untuk tampilan yang lebih rapi
                st.markdown("<hr>", unsafe_allow_html=True)
        else:
            st.write("❌ Tidak ada konten yang tersedia saat ini.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
    finally:
        conn.close()

# Menjalankan aplikasi Streamlit dengan pemanggilan fungsi informasi()
if __name__ == "__main__":
    informasi()
