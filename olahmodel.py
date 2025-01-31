import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import plot_tree
from database import get_connection  # Import koneksi database
import io
# Fungsi untuk mengonversi DataFrame ke format Excel dan mengunduhnya
def to_excel(df):
    # Membuat objek BytesIO untuk menyimpan file Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    output.seek(0)
    return output

def pengelolaan_model():
    st.title("Hasil Model")
    # st.write("Halaman ini digunakan untuk mengelola model yang digunakan dalam aplikasi.")
    
    # Membuat koneksi ke database
    conn = get_connection()
    if conn is None:
        st.error("Koneksi ke database gagal!")
        return

    try:
        cursor = conn.cursor()
        query = "SELECT tahun, provinsi, kabupatenkota, plastik, kertas_karton, logam, kaca FROM datasetbersih"
        cursor.execute(query)
        data = cursor.fetchall()
        df = pd.DataFrame(data, columns=['tahun', 'provinsi', 'kabupatenkota', 'plastik', 'kertas_karton', 'logam', 'kaca'])

        # Proses clustering
        data_clustering = df[['plastik', 'kertas_karton', 'logam', 'kaca']].dropna()
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data_clustering)
        kmeans = KMeans(n_clusters=3, random_state=42)
        data_clustering['cluster'] = kmeans.fit_predict(data_normalized)
        cluster_labels = {0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2'}
        data_clustering['hasil_cluster'] = data_clustering['cluster'].map(cluster_labels)
        df['tahun'] = df['tahun'].apply(lambda x: f"{int(x):d}")  # Menambahkan format integer untuk tahun
        df['hasil_cluster'] = data_clustering['hasil_cluster']

        silhouette_values = silhouette_samples(data_normalized, data_clustering['cluster'])
        data_clustering['silhouette_score'] = silhouette_values
        silhouette_per_cluster = data_clustering.groupby('cluster')['silhouette_score'].mean()
        db_index = davies_bouldin_score(data_normalized, data_clustering['cluster'])
        
        # Menambahkan kolom prediksi nilai ekonomis
        def nilai_ekonomis(cluster):
            if cluster == 'Cluster 0':
                return 'Rendah'
            elif cluster == 'Cluster 1':
                return 'Sedang'
            elif cluster == 'Cluster 2':
                return 'Tinggi'
            else:
                return 'Tidak Dikenal'

        df['prediksi_nilai_ekonomis'] = df['hasil_cluster'].apply(nilai_ekonomis)

        # Tabs untuk hasil
        tab1, tab2, tab3 = st.tabs(["Hasil Clustering", "Hasil Klasifikasi", "Evaluasi Model"])

        with tab1:
            st.subheader("Hasil Clustering")
            st.write(df[['tahun','provinsi', 'kabupatenkota','kertas_karton', 'plastik', 'logam', 'kaca', 'hasil_cluster']])
            silhouette_avg = silhouette_score(data_normalized, data_clustering['cluster'])

            # Tombol download untuk Hasil Clustering
            st.download_button(
                label="Unduh Hasil Clustering (Excel)",
                data=to_excel(df[['provinsi', 'kabupatenkota', 'kertas_karton','plastik', 'logam', 'kaca', 'hasil_cluster']]),
                file_name="hasil_clustering.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # Menambahkan desain kustom untuk tombol
            st.markdown("""
                <style>
                    .stDownloadButton>button {
                        background-color: #dbd28c;
                        color: white;
                        font-size: 16px;
                        font-weight: bold;
                        border: none;
                        padding: 10px 20px;
                        border-radius: 5px;
                        cursor: pointer;
                        transition: background-color 0.3s ease;
                    }

                    .stDownloadButton>button:hover {
                        background-color: #4F4D793;
                    }

                    .stDownloadButton>button:focus {
                        outline: none;
                    }
                </style>
            """, unsafe_allow_html=True)
            st.write(f"Rata-rata Silhouette Score untuk seluruh dataset: {silhouette_avg}")

            # Visualisasi dengan deskripsi cluster, centroid, dan label cluster
            st.subheader("Visualisasi Clustering dengan Deskripsi Cluster")
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(data_clustering['plastik'], data_clustering['logam'], 
                                  c=data_clustering['cluster'], cmap='viridis')
            centroids = kmeans.cluster_centers_
            plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroid')

            # Menambahkan deskripsi warna sesuai cluster
            cluster_labels = ['Rendah', 'Sedang', 'Tinggi']
            colors = plt.cm.viridis(np.linspace(0, 1, 3))
            for i, color in enumerate(colors):
                plt.scatter([], [], c=[color], label=f'Cluster {i}: {cluster_labels[i]}')

            plt.xlabel('Plastik')
            plt.ylabel('Logam')
            plt.title('Clustering: Plastik vs Logam')
            # plt.colorbar(scatter, label='Cluster')
            plt.legend()
            st.pyplot(plt)

            # Menghitung jumlah data per cluster
            cluster_counts = data_clustering['cluster'].value_counts()

            # Menampilkan diagram batang untuk jumlah data per cluster
            st.subheader("Distribusi Jumlah Data per Cluster")
            plt.figure(figsize=(8, 6))
            cluster_counts.plot(kind='bar', color=['#dbd28c', '#4F4D79', '#8cdbd2'])
            plt.xlabel('Cluster')
            plt.ylabel('Jumlah Data')
            plt.title('Distribusi Jumlah Data per Cluster')

            # Menambahkan label untuk setiap cluster dengan rotasi vertikal
            cluster_labels = ['Cluster 0', 'Cluster 1', 'Cluster 2']
            plt.xticks(ticks=range(len(cluster_counts)), labels=cluster_labels, rotation=0, ha='center')

            # Menambahkan jumlah data di atas batang
            for i, v in enumerate(cluster_counts):
                plt.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=12)
            st.pyplot(plt)

        with tab2:
            # Model Decision Tree
            X = df[['kertas_karton','plastik', 'logam', 'kaca']]
            y = df['hasil_cluster']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            clf = DecisionTreeClassifier(random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            st.subheader("Evaluasi Model Klasifikasi (Decision Tree)")
            st.write("Akurasi Model Decision Tree:", accuracy_score(y_test, y_pred))

            df['tahun'] = df['tahun'].apply(lambda x: f"{int(x):d}")  # Menambahkan format integer untuk tahun
            # Menambahkan hasil prediksi ke DataFrame
            df['Cluster'] = clf.predict(X)

            # Fungsi untuk memberikan keterangan nilai ekonomis berdasarkan cluster
            def nilai_ekonomis(cluster):
                if cluster == 'Cluster 0':
                    return 'Rendah'
                elif cluster == 'Cluster 1':
                    return 'Sedang'
                elif cluster == 'Cluster 2':
                    return 'Tinggi'
                else:
                    return 'Tidak Dikenal'

            # Menambahkan kolom prediksi nilai ekonomis
            df['prediksi_nilai_ekonomis'] = df['Cluster'].apply(nilai_ekonomis)

            # Fungsi untuk memberikan keterangan deskriptif terkait nilai ekonomis
            def keterangan_nilai_ekonomis(nilai_ekonomis):
                if nilai_ekonomis == 'Tinggi':
                    return 'Menunjukkan potensi yang sangat menguntungkan dalam pengelolaan sampah.'
                elif nilai_ekonomis == 'Sedang':
                    return 'Potensi ekonomi yang memerlukan pemrosesan lebih lanjut untuk hasil optimal.'
                elif nilai_ekonomis == 'Rendah':
                    return 'Sedikit potensi untuk dimanfaatkan dalam pengelolaan sampah.'
                else:
                    return 'Tidak Dikenal'

            # Menambahkan kolom keterangan nilai ekonomis
            df['keterangan_prediksi'] = df['prediksi_nilai_ekonomis'].apply(keterangan_nilai_ekonomis)

            # Menampilkan tabel dengan provinsi dan kabupaten/kota
            st.write("Hasil klasifikasi dengan kolom Provinsi dan Kabupaten/Kota:")
            hasil_klasifikasi = df[['tahun', 'provinsi', 'kabupatenkota', 'kertas_karton', 'plastik', 'logam', 'kaca', 'Cluster', 'prediksi_nilai_ekonomis', 'keterangan_prediksi']]
            st.write(hasil_klasifikasi)

            # Tombol download untuk Hasil Klasifikasi
            st.download_button(
                label="Unduh Hasil Klasifikasi (Excel)",
                data=to_excel(hasil_klasifikasi),
                file_name="hasil_klasifikasi.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # Menambahkan desain kustom untuk tombol
            st.markdown("""
                <style>
                    .stDownloadButton>button {
                        background-color: #dbd28c;
                        color: white;
                        font-size: 16px;
                        font-weight: bold;
                        border: none;
                        padding: 10px 20px;
                        border-radius: 5px;
                        cursor: pointer;
                        transition: background-color 0.3s ease;
                    }

                    .stDownloadButton>button:hover {
                        background-color: #4F4D793;
                    }

                    .stDownloadButton>button:focus {
                        outline: none;
                    }
                </style>
            """, unsafe_allow_html=True)

            # Diagram batang untuk hasil klasifikasi
            st.subheader("Distribusi Hasil Klasifikasi")
            klasifikasi_counts = df['prediksi_nilai_ekonomis'].value_counts()

            # Membuat diagram batang untuk klasifikasi dengan warna yang berbeda
            plt.figure(figsize=(8, 6))
            bars = klasifikasi_counts.plot(kind='bar', color=['lightcoral', 'lightblue', 'lightgreen'])  # Menggunakan warna berbeda
            plt.title("Distribusi Hasil Klasifikasi Nilai Ekonomis")
            plt.xlabel("Nilai Ekonomis")
            plt.ylabel("Jumlah Data")
            plt.xticks(rotation=0)

            # Menambahkan jumlah data di atas batang
            for i, v in enumerate(klasifikasi_counts):
                plt.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=12)  # Menambahkan angka di atas batang

            st.pyplot(plt)

            # Menampilkan daerah yang termasuk dalam masing-masing prediksi nilai ekonomis
            st.subheader("Daerah yang Termasuk dalam Setiap Klasifikasi Nilai Ekonomis")

            # Menampilkan daerah berdasarkan klasifikasi nilai ekonomis
            for nilai_ekonomis in klasifikasi_counts.index:
                st.write(f"Daerah yang termasuk dalam kategori {nilai_ekonomis} antara lain:")
                daerah_klasifikasi = df[df['prediksi_nilai_ekonomis'] == nilai_ekonomis][['provinsi', 'kabupatenkota']]
                st.write(daerah_klasifikasi)
            
            st.subheader("Visualisasi Klasifikasi")
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            plot_tree(clf, filled=True, feature_names=['kertas_karton', 'plastik', 'logam', 'kaca'], class_names=['Rendah', 'Sedang', 'Tinggi'], rounded=True)
            st.pyplot(fig2)

            st.subheader("Keterangan")
            st.write("""
            **Gini**: Gini index digunakan untuk mengukur kemurnian data pada suatu node. Semakin rendah nilai Gini, semakin murni node tersebut.

            **Samples**: Menunjukkan jumlah data yang ada pada suatu node. 

            **Value**:  Menunjukkan proporsi kelas-kelas yang ada di dalam node tersebut
            """)

            conf_matrix = confusion_matrix(y_test, y_pred)
            fig3, ax3 = plt.subplots(figsize=(3, 2))  # Menyesuaikan ukuran grafik keseluruhan
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Rendah', 'Sedang', 'Tinggi'], yticklabels=['Rendah', 'Sedang', 'Tinggi'], 
                        annot_kws={"size": 6},  # Mengatur ukuran teks angka dalam matriks
                        cbar_kws={'shrink': 0.8})  # Mengatur ukuran colorbar
            ax3.set_xlabel('Prediksi', fontsize=6)  # Mengatur ukuran font untuk label
            ax3.set_ylabel('Aktual', fontsize=6)  # Mengatur ukuran font untuk label
            ax3.set_title('Evaluasi Confusion Matrix', fontsize=12)  # Menambahkan judul dengan ukuran font lebih kecil
            st.pyplot(fig3)

            # Membuat laporan evaluasi
            report = classification_report(y_test, y_pred, target_names=['Rendah', 'Sedang', 'Tinggi'])

            # Menampilkan laporan evaluasi di Streamlit
            st.subheader("Laporan Evaluasi Model")
            st.text(report)

        with tab3:
            st.markdown("### Evaluasi Model :bar_chart:")

            # Menyiapkan data untuk tabel perbandingan evaluasi
            evaluasi_clustering = f"{silhouette_avg:.2f}"
            evaluasi_klasifikasi = f"{accuracy_score(y_test, y_pred):.2f}"

            # Membuat tabel menggunakan HTML
            tabel_html = f"""
            <table style="width: 50%; border: 1px solid black; border-collapse: collapse;">
                <thead>
                    <tr>
                        <th style="padding: 8px; text-align: center; background-color: #3498DB; color: white;">Evaluasi Clustering</th>
                        <th style="padding: 8px; text-align: center; background-color: #3498DB; color: white;">Evaluasi Klasifikasi</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="padding: 8px; text-align: center;">{evaluasi_clustering}</td>
                        <td style="padding: 8px; text-align: center;">{evaluasi_klasifikasi}</td>
                    </tr>
                </tbody>
            </table>
            """
            # Menampilkan tabel menggunakan HTML
            st.markdown(tabel_html, unsafe_allow_html=True)

            
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
    finally:
        conn.close()
