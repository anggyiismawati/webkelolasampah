# olahinformasi.py
import streamlit as st
import mysql.connector
from datetime import datetime
from database import get_connection  # Pastikan get_connection adalah fungsi untuk mendapatkan koneksi ke database

def pengelolaan_informasi():
    """Fungsi untuk halaman Informasi."""
    st.title("Kelola Informasi Website")
    # st.write("Halaman ini berisi informasi terkait pengelolaan sampah, tujuan penelitian, dan lainnya.")

    # Membuat koneksi ke database
    conn = get_connection()
    if conn is None:
        st.error("Koneksi ke database gagal!")
        return

    try:
        cursor = conn.cursor()

        # Form untuk menambahkan konten baru
        with st.expander("Tambah Konten Baru"):
            judul = st.text_input("Judul Konten")
            deskripsi = st.text_area("Deskripsi Konten")
            tanggal = st.date_input("Tanggal", min_value=datetime.today())

            # Upload gambar
            uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:
                # Menyimpan gambar ke direktori dan mendapatkan path
                img_path = f"images/{uploaded_file.name}"
                with open(img_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.image(img_path, caption="Gambar yang diupload", use_container_width=True)

            if st.button("Simpan Konten"):
                if judul and deskripsi and uploaded_file:
                    insert_query = """
                    INSERT INTO konten (judul, deskripsi, tanggal, gambar)
                    VALUES (%s, %s, %s, %s)
                    """
                    cursor.execute(insert_query, (judul, deskripsi, tanggal, img_path))
                    conn.commit()
                    st.success("Konten berhasil ditambahkan!")
                else:
                    st.error("Pastikan semua kolom terisi dengan benar!")

        # Menampilkan daftar konten yang ada untuk dihapus
        with st.expander("Hapus Konten"):
            cursor.execute("SELECT * FROM konten ORDER BY tanggal DESC")
            konten_data = cursor.fetchall()

            if konten_data:
                konten_options = [f"{item[1]} - {item[2]}" for item in konten_data]  # Menampilkan judul dan tanggal
                konten_to_delete = st.selectbox("Pilih konten untuk dihapus", konten_options)

                if konten_to_delete:
                    selected_row = next(item for item in konten_data if f"{item[1]} - {item[2]}" == konten_to_delete)

                    if st.button("Hapus Konten"):
                        delete_query = """
                        DELETE FROM konten WHERE id = %s
                        """
                        cursor.execute(delete_query, (selected_row[0],))  # Menghapus berdasarkan ID
                        conn.commit()
                        st.success("Konten berhasil dihapus!")
            else:
                st.write("Tidak ada konten yang tersedia untuk dihapus.")

        # Menampilkan konten yang sudah diupload di database
        with st.expander("Lihat Konten yang Tersedia"):
            cursor.execute("SELECT judul, deskripsi, tanggal, gambar FROM konten ORDER BY tanggal DESC")
            konten_data = cursor.fetchall()

            if konten_data:
                for konten in konten_data:
                    st.subheader(konten[0])  # Judul Konten
                    st.write(konten[1])  # Deskripsi Konten
                    st.write(f"Tanggal: {konten[2]}")  # Tanggal Konten
                    
                    # Menampilkan gambar dengan judul sebagai caption
                    st.image(konten[3], caption=f"Gambar untuk {konten[0]}", width=400)  # Menggunakan judul sebagai caption
                    
                    # Menambahkan pemisah antar konten
                    st.markdown("---")  # Ini akan menambahkan garis horizontal untuk memisahkan konten

            else:
                st.write("Tidak ada konten yang tersedia.")

    except mysql.connector.Error as err:
        st.error(f"Terjadi kesalahan: {err}")
    finally:
        conn.close()
