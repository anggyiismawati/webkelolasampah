import streamlit as st
import matplotlib.pyplot as plt
import mysql.connector
from database import get_connection

def beranda():
    """Fungsi untuk halaman Beranda."""
    # Menambahkan CSS untuk menyesuaikan font dan warna
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
            .btn-login {
                background-color: #3498db;
                color: white;
                border-radius: 5px;
                padding: 12px 20px;
                font-size: 18px;
                border: none;
                cursor: pointer;
                font-family: 'Arial', sans-serif;
            }
            .btn-login:hover {
                background-color: #2980b9;
            }
            .header {
                background-color: #ecf0f1;
                padding: 20px;
                border-radius: 10px;
            }
            .container {
                background-color: white;
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            }
            .footer {
                color: #7f8c8d;
                text-align: center;
                font-size: 14px;
                margin-top: 20px;
                font-family: 'Verdana', sans-serif;
            }
            .subheader {
                color: #16a085;
                font-size: 20px;
            }
            .call-to-action {
                background-color: #2ecc71;
                color: white;
                padding: 12px 25px;
                border-radius: 5px;
                font-size: 18px;
                cursor: pointer;
                font-family: 'Arial', sans-serif;
            }
            .call-to-action:hover {
                background-color: #27ae60;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Selamat datang di halaman beranda!")
    
    # Menambahkan deskripsi singkat tentang aplikasi dengan lebih interaktif
    st.subheader("Tentang Aplikasi")
    st.markdown("""
    **Aplikasi ini digunakan untuk mengelola data pengelolaan sampah** berdasarkan nilai ekonomis. 
    Anda dapat login untuk mengakses data lebih lanjut dan melakukan analisis.
    """)
    
    # Kolom untuk menampilkan statistik
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Mengambil data dari database
        conn = get_connection()
        if conn is None:
            st.error("Koneksi ke database gagal!")
            return
        
        try:
            cursor = conn.cursor()
            
            # Ambil data komposisi sampah dari tabel datasetbersih
            query = "SELECT sisa_makanan, plastik, kain, karet_kulit, logam, kertas_karton, kayu_ranting, kaca, lainnya FROM datasetbersih LIMIT 1"
            cursor.execute(query)
            data = cursor.fetchone()

            # Jika data ditemukan, tampilkan
            if data:
                categories = ['Sisa Makanan','Plastik', 'Logam', 'Kertas Karton', 'Kayu Ranting', 'Kaca', 'Kain', 'Karet Kulit', 'Lainnya']
                values = list(data)

                # Menambahkan grafik komposisi sampah
                st.subheader("Statistik Sampah")
                st.write("Berikut adalah statistik pengelolaan sampah berdasarkan data yang ada di database:")

                fig, ax = plt.subplots()
                ax.pie(values, labels=categories, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0','#ffb3e6','#ff6666','#c2f0c2','#b3b3cc'])
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig)
            else:
                st.write("Data komposisi sampah tidak ditemukan.")
        
        except Exception as e:
            st.error(f"Terjadi kesalahan saat mengambil data: {e}")
        
        finally:
            conn.close()

    # Kolom 2: Memberikan informasi lebih lanjut atau tips
    with col2:
        st.subheader("Pentingnya Pengelolaan Sampah")
        st.write("""
        Pengelolaan sampah yang efektif sangat penting untuk meningkatkan kualitas lingkungan. 
        Dengan memanfaatkan sistem ini, Anda dapat mengetahui kategori sampah yang perlu dikelola dengan lebih baik.
        """)

    # Tips tambahan atau call to action dengan desain footer
    st.markdown("""
    **Ayo, mulai kelola sampah dengan bijak!** 
    Anda bisa memanfaatkan data ini untuk membantu pengelolaan sampah yang lebih baik dan berkelanjutan.
    """)
    
    # Footer untuk informasi tambahan
    st.markdown("<div class='footer'>© 2024 Pengelolaan Sampah Berkelanjutan</div>", unsafe_allow_html=True)
