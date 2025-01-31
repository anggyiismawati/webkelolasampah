import streamlit as st
import pandas as pd
import mysql.connector
from database import get_connection  # Pastikan get_connection ada di database.py

def upload_data():
    """Fungsi untuk halaman upload data dari file Excel."""
    st.title("Upload Data")

    # Upload file Excel
    uploaded_file = st.file_uploader("Pilih file Excel", type="xlsx")
    
    # Mengecek jika file sudah di-upload
    if uploaded_file:
        # Membaca file Excel menggunakan pandas
        try:
            # Membaca file Excel
            df = pd.read_excel(uploaded_file)

            # Menampilkan data yang di-upload untuk konfirmasi
            st.write("Berikut adalah data yang akan di-upload:")
            # Menampilkan kolom tahun tanpa koma
            df['tahun'] = df['tahun'].apply(lambda x: f"{int(x):d}")  # Menambahkan format integer untuk tahun
            st.dataframe(df)

            # Menambahkan tombol untuk meng-upload data
            if st.button("Upload Data"):
                # Membuat koneksi ke database
                conn = get_connection()
                if conn is None:
                    st.error("Koneksi ke database gagal!")
                    return

                try:
                    # Membuat cursor untuk melakukan query
                    cursor = conn.cursor()

                    # Menyimpan query yang akan dieksekusi
                    queries_to_execute = []

                    # Memproses setiap baris dalam dataset
                    for index, row in df.iterrows():
                        # Mengecek apakah data dengan tahun yang sama sudah ada
                        select_query = """
                        SELECT COUNT(*) FROM datasetbersih WHERE tahun = %s
                        """
                        cursor.execute(select_query, (row['tahun'],))
                        count = cursor.fetchone()[0]

                        if count > 0:
                            # Jika data untuk tahun tersebut sudah ada, lakukan update
                            update_query = """
                            UPDATE datasetbersih SET provinsi = %s, kabupatenkota = %s, sisa_makanan = %s, kayu_ranting = %s,
                            kertas_karton = %s, plastik = %s, logam = %s, kain = %s, karet_kulit = %s, kaca = %s, lainnya = %s
                            WHERE tahun = %s
                            """
                            queries_to_execute.append((update_query, (
                                row['provinsi'], row['kabupatenkota'], row['sisa_makanan'], row['kayu_ranting'],
                                row['kertas_karton'], row['plastik'], row['logam'], row['kain'], row['karet_kulit'],
                                row['kaca'], row['lainnya'], row['tahun']
                            )))
                        else:
                            # Jika data untuk tahun tersebut belum ada, lakukan insert
                            insert_query = """
                            INSERT INTO datasetbersih (tahun, provinsi, kabupatenkota, sisa_makanan, kayu_ranting,
                                                       kertas_karton, plastik, logam, kain, karet_kulit, kaca, lainnya)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """
                            queries_to_execute.append((insert_query, (
                                row['tahun'], row['provinsi'], row['kabupatenkota'], row['sisa_makanan'], row['kayu_ranting'],
                                row['kertas_karton'], row['plastik'], row['logam'], row['kain'], row['karet_kulit'],
                                row['kaca'], row['lainnya']
                            )))

                    # Menjalankan semua query dalam satu commit untuk efisiensi
                    for query, params in queries_to_execute:
                        cursor.execute(query, params)

                    # Commit perubahan ke database
                    conn.commit()
                    st.success("Data berhasil di-upload!")

                except mysql.connector.Error as err:
                    st.error(f"Terjadi kesalahan: {err}")
                finally:
                    conn.close()

        except Exception as e:
            st.error(f"Gagal memproses file Excel: {e}")
