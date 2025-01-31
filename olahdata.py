import streamlit as st
import mysql.connector
from database import get_connection  # Pastikan fungsi get_connection ada di database.py
from upload_data import upload_data  # Pastikan ini benar


def pengelolaan_data():
    """Fungsi untuk halaman pengelolaan data."""
    st.title("Pengelolaan Data")

     # Menambahkan menu untuk upload data Excel
    with st.expander("Upload Data Excel"):
        upload_data()  # Memanggil fungsi untuk upload data

    # Membuat koneksi ke database
    conn = get_connection()
    if conn is None:
        st.error("Koneksi ke database gagal!")
        return

    try:
        # Membuat cursor dan mengeksekusi query untuk mengambil data
        cursor = conn.cursor(dictionary=True)  # dictionary=True agar hasil query berupa dictionary
        query = """
        SELECT tahun, provinsi, kabupatenkota, sisa_makanan, kayu_ranting, kertas_karton,
               plastik, logam, kain, karet_kulit, kaca, lainnya
        FROM datasetbersih
        """
        cursor.execute(query)
        data = cursor.fetchall()  # Mengambil semua data dari query

        if data:
            # Menampilkan data dalam bentuk tabel
            st.write("Berikut adalah data komposisi sampah:")
            st.dataframe([{
                **row, 
                'tahun': f"{int(row['tahun']):d}"  # Menampilkan tahun sebagai integer tanpa koma
            } for row in data])  # Menampilkan data dalam bentuk dataframe
        else:
            st.write("Tidak ada data yang ditemukan.")

        # Kotak untuk operasi CRUD
        with st.expander("Tambah Data Baru"):
            tahun = st.number_input("Tahun", min_value=2000, max_value=2100, format="%d")  # Pastikan formatnya integer
            provinsi = st.text_input("Provinsi")
            kabupatenkota = st.text_input("Kabupaten/Kota")
            sisa_makanan = st.number_input("Sisa Makanan (%)", min_value=0.0, max_value=100.0, step=0.1)
            kayu_ranting = st.number_input("Kayu Ranting (%)", min_value=0.0, max_value=100.0, step=0.1)
            kertas_karton = st.number_input("Kertas Karton (%)", min_value=0.0, max_value=100.0, step=0.1)
            plastik = st.number_input("Plastik (%)", min_value=0.0, max_value=100.0, step=0.1)
            logam = st.number_input("Logam (%)", min_value=0.0, max_value=100.0, step=0.1)
            kain = st.number_input("Kain (%)", min_value=0.0, max_value=100.0, step=0.1)
            karet_kulit = st.number_input("Karet Kulit (%)", min_value=0.0, max_value=100.0, step=0.1)
            kaca = st.number_input("Kaca (%)", min_value=0.0, max_value=100.0, step=0.1)
            lainnya = st.number_input("Lainnya (%)", min_value=0.0, max_value=100.0, step=0.1)

            if st.button("Simpan Data Baru"):
                # Insert data baru ke dalam database
                insert_query = """
                INSERT INTO datasetbersih (tahun, provinsi, kabupatenkota, sisa_makanan, kayu_ranting,
                                           kertas_karton, plastik, logam, kain, karet_kulit, kaca, lainnya)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(insert_query, (tahun, provinsi, kabupatenkota, sisa_makanan, kayu_ranting,
                                              kertas_karton, plastik, logam, kain, karet_kulit, kaca, lainnya))
                conn.commit()
                st.success("Data berhasil ditambahkan!")

        with st.expander("Update Data"):
            data_for_update = st.selectbox("Pilih data untuk diupdate", [f"{row['tahun']} - {row['provinsi']} - {row['kabupatenkota']}" for row in data])

            if data_for_update:
                selected_row = next(item for item in data if f"{item['tahun']} - {item['provinsi']} - {item['kabupatenkota']}" == data_for_update)

                tahun_update = st.number_input("Tahun", value=selected_row['tahun'], min_value=2000, max_value=2100, format="%d")  # Pastikan formatnya integer
                provinsi_update = st.text_input("Provinsi", value=selected_row['provinsi'])
                kabupatenkota_update = st.text_input("Kabupaten/Kota", value=selected_row['kabupatenkota'])
                sisa_makanan_update = st.number_input("Sisa Makanan (%)", value=selected_row['sisa_makanan'], min_value=0.0, max_value=100.0, step=0.1)
                kayu_ranting_update = st.number_input("Kayu Ranting (%)", value=selected_row['kayu_ranting'], min_value=0.0, max_value=100.0, step=0.1)
                kertas_karton_update = st.number_input("Kertas Karton (%)", value=selected_row['kertas_karton'], min_value=0.0, max_value=100.0, step=0.1)
                plastik_update = st.number_input("Plastik (%)", value=selected_row['plastik'], min_value=0.0, max_value=100.0, step=0.1)
                logam_update = st.number_input("Logam (%)", value=selected_row['logam'], min_value=0.0, max_value=100.0, step=0.1)
                kain_update = st.number_input("Kain (%)", value=selected_row['kain'], min_value=0.0, max_value=100.0, step=0.1)
                karet_kulit_update = st.number_input("Karet Kulit (%)", value=selected_row['karet_kulit'], min_value=0.0, max_value=100.0, step=0.1)
                kaca_update = st.number_input("Kaca (%)", value=selected_row['kaca'], min_value=0.0, max_value=100.0, step=0.1)
                lainnya_update = st.number_input("Lainnya (%)", value=selected_row['lainnya'], min_value=0.0, max_value=100.0, step=0.1)

                if st.button("Update Data"):
                    update_query = """
                    UPDATE datasetbersih SET tahun = %s, provinsi = %s, kabupatenkota = %s, sisa_makanan = %s,
                    kayu_ranting = %s, kertas_karton = %s, plastik = %s, logam = %s, kain = %s, karet_kulit = %s,
                    kaca = %s, lainnya = %s WHERE tahun = %s AND provinsi = %s AND kabupatenkota = %s
                    """
                    cursor.execute(update_query, (tahun_update, provinsi_update, kabupatenkota_update, sisa_makanan_update,
                                                  kayu_ranting_update, kertas_karton_update, plastik_update, logam_update,
                                                  kain_update, karet_kulit_update, kaca_update, lainnya_update,
                                                  selected_row['tahun'], selected_row['provinsi'], selected_row['kabupatenkota']))
                    conn.commit()
                    st.success("Data berhasil diperbarui!")

        with st.expander("Hapus Data Per-wilayah"):
            data_for_delete = st.selectbox("Hapus data per-wilayah", [f"{row['tahun']} - {row['provinsi']} - {row['kabupatenkota']}" for row in data])

            if data_for_delete:
                selected_row_to_delete = next(item for item in data if f"{item['tahun']} - {item['provinsi']} - {item['kabupatenkota']}" == data_for_delete)
                if st.button("Hapus Data"):
                    delete_query = """
                    DELETE FROM datasetbersih WHERE tahun = %s AND provinsi = %s AND kabupatenkota = %s
                    """
                    cursor.execute(delete_query, (selected_row_to_delete['tahun'], selected_row_to_delete['provinsi'],
                                                  selected_row_to_delete['kabupatenkota']))
                    conn.commit()
                    st.success("Data berhasil dihapus!")
            
        # Hapus Data Per-Tahun
        with st.expander("Hapus Data Per-Tahun"):
            data_for_delete_year = st.selectbox("Hapus data per-tahun", sorted(set(row['tahun'] for row in data)))

            if data_for_delete_year:
                if st.button("Hapus Data Per Tahun"):
                    delete_query = """
                    DELETE FROM datasetbersih WHERE tahun = %s
                    """
                    cursor.execute(delete_query, (data_for_delete_year,))
                    conn.commit()
                    st.success("Data untuk tahun tersebut berhasil dihapus!")

    except mysql.connector.Error as err:
        st.error(f"Terjadi kesalahan: {err}")
    finally:
        conn.close()
