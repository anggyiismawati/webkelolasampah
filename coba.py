# Menambahkan contoh daerah untuk setiap kategori
#             kategori_daerah = hasil_klasifikasi.groupby('prediksi_nilai_ekonomis')['kabupatenkota'].apply(list)

#             for kategori, daerah in kategori_daerah.items():
#                 st.write(f"**Kategori {kategori}:**")
#                 # st.write(", ".join(daerah))
#                 st.write("\n".join([f"- {d}" for d in daerah]))