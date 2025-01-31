-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Waktu pembuatan: 31 Jan 2025 pada 12.01
-- Versi server: 10.4.32-MariaDB
-- Versi PHP: 8.0.30

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `kelola_sampah`
--

-- --------------------------------------------------------

--
-- Struktur dari tabel `datasetbersih`
--

CREATE TABLE `datasetbersih` (
  `id_wilayah` int(11) NOT NULL,
  `tahun` int(11) DEFAULT NULL,
  `provinsi` varchar(100) DEFAULT NULL,
  `kabupatenkota` varchar(100) DEFAULT NULL,
  `sisa_makanan` float DEFAULT NULL,
  `kayu_ranting` float DEFAULT NULL,
  `kertas_karton` float DEFAULT NULL,
  `plastik` float DEFAULT NULL,
  `logam` float DEFAULT NULL,
  `kain` float DEFAULT NULL,
  `karet_kulit` float DEFAULT NULL,
  `kaca` float DEFAULT NULL,
  `lainnya` float DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data untuk tabel `datasetbersih`
--

INSERT INTO `datasetbersih` (`id_wilayah`, `tahun`, `provinsi`, `kabupatenkota`, `sisa_makanan`, `kayu_ranting`, `kertas_karton`, `plastik`, `logam`, `kain`, `karet_kulit`, `kaca`, `lainnya`) VALUES
(584, 2023, 'Aceh', 'Kab. Aceh Selatan', 30.08, 12.29, 13.1, 16.33, 6.4, 5.42, 3.43, 5.35, 7.6),
(585, 2023, 'Aceh', 'Kab. Aceh Timur', 37, 14, 5, 26, 5, 3, 2, 4, 4),
(586, 2023, 'Aceh', 'Kab. Aceh Tengah', 22, 18, 7, 26, 3, 11, 5, 5, 3),
(587, 2023, 'Aceh', 'Kab. Aceh Barat', 48, 10, 7, 22, 4, 3, 2, 2, 2),
(588, 2023, 'Aceh', 'Kab. Aceh Besar', 25.5, 25.5, 9.5, 21.6, 5.1, 3.3, 3.3, 3.7, 2.5),
(589, 2023, 'Aceh', 'Kab. Pidie', 8.5, 7.5, 19.3, 35.7, 4, 1, 3, 2, 19),
(590, 2023, 'Aceh', 'Kab. Aceh Utara', 30, 10, 10, 40, 2, 2, 2, 1, 3),
(591, 2023, 'Aceh', 'Kab. Simeulue', 22.8, 15.3, 12.1, 26.2, 10.2, 2.6, 2.8, 2, 6),
(592, 2023, 'Aceh', 'Kab. Aceh Singkil', 30, 10, 15, 22, 2, 5, 3, 2, 11),
(593, 2023, 'Aceh', 'Kab. Bireuen', 41.1, 10.2, 15.2, 19.2, 2.7, 2.2, 2.1, 2.1, 5.2),
(594, 2023, 'Aceh', 'Kab. Aceh Barat Daya', 32.8, 9.5, 20, 12, 10, 2.2, 3.5, 5, 5),
(595, 2023, 'Aceh', 'Kab. Gayo Lues', 40, 17, 8, 20, 5, 4, 1, 3, 2),
(596, 2023, 'Aceh', 'Kab. Aceh Jaya', 45, 14, 11, 15, 8, 2, 2, 1, 2),
(597, 2023, 'Aceh', 'Kab. Nagan Raya', 42, 10, 5, 34, 1, 1, 1, 1, 5),
(598, 2023, 'Aceh', 'Kab. Aceh Tamiang', 30, 2, 5, 55, 3, 1, 2, 1, 1),
(599, 2023, 'Aceh', 'Kab. Bener Meriah', 21, 25, 19, 18, 3, 3, 3, 3, 5),
(600, 2023, 'Aceh', 'Kab. Pidie Jaya', 20, 27, 14, 26, 2, 3, 3, 3, 2),
(601, 2023, 'Aceh', 'Kota Banda Aceh', 39.4, 20.6, 3, 11.8, 4.6, 3.8, 4.2, 4.75, 7.85),
(602, 2023, 'Aceh', 'Kota Sabang', 20, 10, 5, 20, 10, 5, 5, 15, 10),
(603, 2023, 'Aceh', 'Kota Lhokseumawe', 54.55, 16.31, 0.2, 14.1, 1.82, 0.13, 0.52, 2.45, 9.92),
(604, 2023, 'Aceh', 'Kota Langsa', 22, 20, 9, 15, 7, 10, 7, 8, 2),
(605, 2023, 'Aceh', 'Kota Subulussalam', 26, 20, 10, 12, 2, 8, 5, 2, 15),
(606, 2023, 'Sumatera Utara', 'Kab. Langkat', 38, 30, 24, 7, 0.2, 0.2, 0.1, 0.1, 0.1),
(607, 2023, 'Sumatera Utara', 'Kab. Deli Serdang', 20.5, 24, 13.8, 20.2, 6, 5, 3, 2, 5.5),
(608, 2023, 'Sumatera Utara', 'Kab. Simalungun', 5.5, 1.93, 15.44, 20.46, 0.37, 7.7, 4.5, 8.6, 35.5),
(609, 2023, 'Sumatera Utara', 'Kab. Labuhanbatu', 25, 24, 4, 30, 2, 6, 2, 4, 3),
(610, 2023, 'Sumatera Utara', 'Kab. Toba', 25.15, 13.4, 4.13, 9.39, 1.38, 4.56, 13.25, 6.45, 22.29),
(611, 2023, 'Sumatera Utara', 'Kab. Humbang Hasundutan', 12, 55, 10, 13, 3, 1, 0.2, 0.8, 5),
(612, 2023, 'Sumatera Utara', 'Kota Pematang Siantar', 45, 6, 12, 10, 5, 3, 4, 4, 11),
(613, 2023, 'Sumatera Barat', 'Kab. Solok', 29, 20, 12, 22, 2, 2, 2, 2, 9),
(614, 2023, 'Sumatera Barat', 'Kab. Padang Pariaman', 31.82, 0.25, 12.78, 43.77, 0.21, 0.78, 2.07, 0.5, 7.82),
(615, 2023, 'Sumatera Barat', 'Kab. Agam', 70.44, 4.02, 6.08, 8.15, 1.9, 0.28, 0.1, 0.55, 8.48),
(616, 2023, 'Sumatera Barat', 'Kab. Pasaman', 66.28, 2.1, 11.41, 11.25, 2.93, 1.41, 1.36, 1.86, 1.4),
(617, 2023, 'Sumatera Barat', 'Kab. Solok Selatan', 34.35, 0.78, 16.35, 25.69, 0.13, 0.59, 0.55, 2.31, 19.25),
(618, 2023, 'Sumatera Barat', 'Kota Padang', 63, 6.64, 6, 12.4, 0.16, 3, 1.5, 2.4, 4.9),
(619, 2023, 'Sumatera Barat', 'Kota Solok', 45.73, 11.12, 16.88, 20.72, 1.78, 0.35, 0.14, 0.81, 2.47),
(620, 2023, 'Sumatera Barat', 'Kota Padang Panjang', 43.15, 23.56, 14.35, 12.67, 1.5, 0.95, 0.8, 1.19, 1.83),
(621, 2023, 'Sumatera Barat', 'Kota Payakumbuh', 48.25, 9.81, 15.45, 17.47, 2.01, 1.17, 0.45, 0.89, 4.5),
(622, 2023, 'Sumatera Barat', 'Kota Pariaman', 54.18, 10.95, 14.56, 13.77, 1.66, 0.91, 0.65, 1.61, 1.71),
(623, 2023, 'Riau', 'Kab. Indragiri Hulu', 30, 15, 5, 20, 5, 5, 5, 5, 10),
(624, 2023, 'Riau', 'Kab. Bengkalis', 16.14, 10.96, 6.79, 9.47, 1.05, 4.78, 5.12, 13.11, 32.58),
(625, 2023, 'Riau', 'Kab. Rokan Hilir', 14, 22, 15, 32, 3, 4, 2, 2, 6),
(626, 2023, 'Riau', 'Kab. Siak', 39, 15, 10, 26, 1, 2, 2, 3, 2),
(627, 2023, 'Riau', 'Kota Dumai', 48.97, 7.5, 5.3, 8.95, 2.27, 3.72, 2.5, 2.79, 18),
(628, 2023, 'Jambi', 'Kab. Bungo', 60.5, 4.1, 14.1, 5.9, 1.8, 1.5, 5.15, 1, 5.95),
(629, 2023, 'Jambi', 'Kab. Tebo', 62.44, 4.6, 14.3, 5.86, 1.7, 1.3, 4.6, 1, 4.2),
(630, 2023, 'Jambi', 'Kota Jambi', 36.65, 21.31, 9.44, 18.99, 2.06, 10.82, 0.3, 0.28, 0.14),
(631, 2023, 'Sumatera Selatan', 'Kab. Muara Enim', 61.02, 8.07, 7.38, 17.18, 1.14, 1.33, 0.36, 0.31, 3.21),
(632, 2023, 'Sumatera Selatan', 'Kab. Lahat', 60, 10.1, 2, 14, 1.74, 5.5, 0.26, 0.9, 5.5),
(633, 2023, 'Sumatera Selatan', 'Kab. Ogan Komering Ulu Timur', 12, 9, 22, 27, 3, 5, 3, 2, 17),
(634, 2023, 'Sumatera Selatan', 'Kab. Ogan Ilir', 53.63, 0.76, 5.75, 21.63, 0.82, 1.94, 1.82, 1.99, 11.66),
(635, 2023, 'Sumatera Selatan', 'Kab. Musi Rawas Utara', 42, 6, 11, 16, 5, 5, 8, 4, 3),
(636, 2023, 'Sumatera Selatan', 'Kota Pagar Alam', 48.98, 6.4, 6.47, 20.38, 2.2, 4.27, 2.7, 1.2, 7.4),
(637, 2023, 'Sumatera Selatan', 'Kota Prabumulih', 53, 3, 14.8, 25.2, 1, 0.25, 0.25, 2, 0.5),
(638, 2023, 'Bengkulu', 'Kab. Rejang Lebong', 14, 12, 11, 16, 11, 9, 8, 8, 11),
(639, 2023, 'Bengkulu', 'Kab. Bengkulu Utara', 31.6, 37.5, 5.4, 14.2, 3.1, 0.8, 1.2, 2.5, 3.7),
(640, 2023, 'Bengkulu', 'Kab. Seluma', 76.5, 5.4, 6.9, 5.6, 2.3, 2.1, 0.3, 0.4, 0.5),
(641, 2023, 'Bengkulu', 'Kab. Lebong', 64.45, 8.57, 8.56, 12.86, 0.2, 0.89, 1.6, 0.07, 2.8),
(642, 2023, 'Bengkulu', 'Kota Bengkulu', 65, 3, 13, 11, 1, 1, 1, 1, 4),
(643, 2023, 'Lampung', 'Kab. Lampung Tengah', 41, 1, 2, 24, 1, 19, 8, 3, 1),
(644, 2023, 'Lampung', 'Kab. Tanggamus', 26.58, 12.89, 8.81, 30.61, 6.16, 1.76, 6.19, 5.71, 0.99),
(645, 2023, 'Lampung', 'Kab. Pesawaran', 31, 16, 9, 16, 15, 3, 2, 0.2, 7.8),
(646, 2023, 'Lampung', 'Kab. Pringsewu', 50, 13, 5, 20, 1.75, 3, 1, 0.25, 6),
(647, 2023, 'Lampung', 'Kab. Pesisir Barat', 42.5, 19.18, 9.5, 21.1, 0.57, 2.03, 0.89, 1.19, 3.04),
(648, 2023, 'Lampung', 'Kota Bandar Lampung', 15.8, 20.1, 3, 2.6, 1.7, 1.7, 2.7, 1.7, 50.7),
(649, 2023, 'Lampung', 'Kota Metro', 56.47, 6.24, 9.4, 24.07, 0.56, 1.25, 0.27, 0.88, 0.86),
(650, 2023, 'Kepulauan Bangka Belitung', 'Kab. Bangka', 39.85, 13.96, 12.07, 16.97, 3.37, 2.69, 1.98, 2.31, 6.8),
(651, 2023, 'Kepulauan Bangka Belitung', 'Kab. Bangka Barat', 40, 6, 24, 17, 1.1, 1.5, 3.5, 2.6, 4.3),
(652, 2023, 'Kepulauan Riau', 'Kab. Bintan', 41.76, 10.12, 13.2, 21.65, 2.8, 2.1, 1.06, 3.31, 4),
(653, 2023, 'Kepulauan Riau', 'Kab. Karimun', 39.91, 12.51, 11.33, 18.75, 3.05, 2.66, 2.17, 2.48, 7.14),
(654, 2023, 'DKI Jakarta', 'Kota Adm. Jakarta Selatan', 49.87, 3.18, 17.24, 22.95, 1.08, 0.9, 0.7, 1.48, 2.6),
(655, 2023, 'Jawa Barat', 'Kab. Ciamis', 38, 0.4, 16, 20, 10, 4, 1, 10.2, 0.4),
(656, 2023, 'Jawa Barat', 'Kab. Kuningan', 40, 17, 10, 17, 6, 2, 3, 2, 3),
(657, 2023, 'Jawa Barat', 'Kab. Sumedang', 30, 7.5, 10.5, 27.5, 3, 4.5, 2.5, 6.5, 8),
(658, 2023, 'Jawa Barat', 'Kab. Indramayu', 27, 6.4, 25, 35.8, 1, 1.8, 0.8, 0.7, 1.5),
(659, 2023, 'Jawa Barat', 'Kota Bogor', 40, 14, 20, 15, 1, 2.5, 0.5, 1, 6),
(660, 2023, 'Jawa Barat', 'Kota Sukabumi', 40.02, 14.4, 15, 17, 3, 3, 2, 2, 3.58),
(661, 2023, 'Jawa Barat', 'Kota Bandung', 44.52, 3.98, 13.12, 16.7, 0.9, 4.75, 2.38, 1.97, 11.68),
(662, 2023, 'Jawa Barat', 'Kota Bekasi', 65.2, 7.2, 4.2, 15.6, 0.2, 6.3, 0.5, 0.5, 0.3),
(663, 2023, 'Jawa Tengah', 'Kab. Purworejo', 15.2, 52.6, 2.3, 14.4, 1.2, 1.4, 0.1, 1.4, 11.4),
(664, 2023, 'Jawa Tengah', 'Kab. Wonosobo', 68.93, 1.5, 12.57, 10.11, 3.5, 0.6, 0.31, 1.14, 1.34),
(665, 2023, 'Jawa Tengah', 'Kab. Magelang', 45.41, 6.71, 4.43, 24.87, 0.61, 2.15, 0.68, 1.32, 13.82),
(666, 2023, 'Jawa Tengah', 'Kab. Boyolali', 32.18, 11.85, 9.37, 37.47, 5.39, 0.67, 0.7, 0.7, 1.67),
(667, 2023, 'Jawa Tengah', 'Kab. Klaten', 60, 5, 5, 20, 0.5, 5, 3, 0.5, 1),
(668, 2023, 'Jawa Tengah', 'Kab. Sukoharjo', 52.55, 0.42, 13.33, 23.79, 0.89, 1.19, 0.33, 1.03, 6.47),
(669, 2023, 'Jawa Tengah', 'Kab. Wonogiri', 25, 10, 20, 25, 5, 5, 2.5, 5, 2.5),
(670, 2023, 'Jawa Tengah', 'Kab. Sragen', 74.3, 0.8, 8.9, 11.2, 0.5, 1.8, 0.3, 1.8, 0.4),
(671, 2023, 'Jawa Tengah', 'Kab. Blora', 38, 23, 10, 15, 7, 2, 3, 1.5, 0.5),
(672, 2023, 'Jawa Tengah', 'Kab. Rembang', 38.12, 1.59, 13.38, 34.45, 0.93, 1.88, 0.77, 1.33, 7.55),
(673, 2023, 'Jawa Tengah', 'Kab. Pati', 30, 5, 20, 25, 5, 5, 2, 5, 3),
(674, 2023, 'Jawa Tengah', 'Kab. Kudus', 29.65, 16.89, 19.08, 13.02, 5.24, 5.43, 3.62, 5.77, 1.3),
(675, 2023, 'Jawa Tengah', 'Kab. Jepara', 53.31, 0.96, 8.79, 23.36, 0.69, 2.44, 0.94, 1.16, 8.35),
(676, 2023, 'Jawa Tengah', 'Kab. Demak', 48.54, 4.59, 16.31, 22.54, 0.06, 0.05, 3.75, 2.8, 1.36),
(677, 2023, 'Jawa Tengah', 'Kab. Temanggung', 31.33, 17.62, 6.68, 18.27, 7.41, 1.96, 2.02, 2.06, 12.65),
(678, 2023, 'Jawa Tengah', 'Kab. Kendal', 30.61, 15.08, 9.49, 23.02, 0.29, 8, 1.07, 0.88, 11.56),
(679, 2023, 'Jawa Tengah', 'Kab. Pekalongan', 7, 8.2, 6.5, 10.2, 0.8, 3, 2, 0.2, 62.1),
(680, 2023, 'Jawa Tengah', 'Kab. Tegal', 37, 12, 15, 24, 3, 2, 2, 1, 4),
(681, 2023, 'Jawa Tengah', 'Kab. Brebes', 5, 50, 5, 25, 4, 4, 1, 1, 5),
(682, 2023, 'Jawa Tengah', 'Kota Magelang', 41.17, 7.54, 15.93, 23.28, 3.88, 0.03, 0.19, 1.89, 6.09),
(683, 2023, 'D.I. Yogyakarta', 'Kota Yogyakarta', 46.45, 0.25, 17.4, 31.05, 0.55, 1.05, 0.6, 1.05, 1.6),
(684, 2023, 'Jawa Timur', 'Kab. Pacitan', 68.2, 2.9, 8.2, 17.2, 0.3, 0.2, 0.3, 0.5, 2.2),
(685, 2023, 'Jawa Timur', 'Kab. Trenggalek', 67.48, 3.55, 10.77, 11.62, 1, 1.05, 1.14, 1.6, 1.79),
(686, 2023, 'Jawa Timur', 'Kab. Tulungagung', 57.3, 7.3, 10.6, 12.6, 4, 1.7, 3, 1.1, 2.4),
(687, 2023, 'Jawa Timur', 'Kab. Kediri', 46.4, 22.47, 4.28, 11.31, 0.2, 2.02, 0.63, 4.75, 7.94),
(688, 2023, 'Jawa Timur', 'Kab. Lumajang', 43.3, 16.25, 9.3, 15.3, 1.17, 2.25, 2.3, 1.3, 8.83),
(689, 2023, 'Jawa Timur', 'Kab. Situbondo', 44.4, 13.8, 16.2, 24, 0.2, 0.6, 0.2, 0.2, 0.4),
(690, 2023, 'Jawa Timur', 'Kab. Probolinggo', 20, 32, 16.6, 16.9, 0.4, 0.5, 12, 0.2, 1.4),
(691, 2023, 'Jawa Timur', 'Kab. Mojokerto', 47.88, 1.91, 19.54, 21.07, 0.53, 0.47, 0.17, 0.24, 8.19),
(692, 2023, 'Jawa Timur', 'Kab. Jombang', 49, 12, 8, 15, 1.8, 5, 1.6, 1.6, 6),
(693, 2023, 'Jawa Timur', 'Kab. Nganjuk', 61.5, 2, 13.1, 12.3, 6, 1, 1, 1, 2.1),
(694, 2023, 'Jawa Timur', 'Kab. Madiun', 39.8, 11.3, 13.3, 19.7, 5.34, 2.65, 2.02, 4.91, 0.98),
(695, 2023, 'Jawa Timur', 'Kab. Magetan', 35.23, 24.05, 9.18, 15.65, 0.45, 1.12, 1.18, 2.88, 10.26),
(696, 2023, 'Jawa Timur', 'Kab. Ngawi', 39.4, 4.9, 5.5, 32.5, 2.2, 1.5, 4.4, 3.7, 5.9),
(697, 2023, 'Jawa Timur', 'Kab. Tuban', 58.52, 15.32, 5.78, 11.02, 0.17, 1.62, 0.1, 1.33, 6.14),
(698, 2023, 'Jawa Timur', 'Kab. Lamongan', 30, 12, 9, 18, 8, 2, 2, 1, 18),
(699, 2023, 'Jawa Timur', 'Kab. Gresik', 55.29, 10.29, 4.47, 11.58, 0.47, 4.84, 2.02, 1.37, 9.67),
(700, 2023, 'Jawa Timur', 'Kab. Bangkalan', 52.32, 15.32, 9.72, 14.97, 0.43, 1.02, 0.29, 2.93, 3),
(701, 2023, 'Jawa Timur', 'Kab. Sampang', 45, 11, 10, 15, 1, 2, 1, 1, 14),
(702, 2023, 'Jawa Timur', 'Kab. Pamekasan', 55, 4, 7, 13, 3, 5, 2, 2, 9),
(703, 2023, 'Jawa Timur', 'Kota Kediri', 58, 7, 12.2, 13.4, 3.4, 1.4, 1.6, 0.8, 2.2),
(704, 2023, 'Jawa Timur', 'Kota Blitar', 42.4, 13.2, 8.56, 16.5, 3.8, 2.6, 2.85, 2.94, 7.15),
(705, 2023, 'Jawa Timur', 'Kota Malang', 54.39, 13.6, 4.47, 13.66, 0.98, 0.52, 0.21, 1.78, 10.39),
(706, 2023, 'Jawa Timur', 'Kota Probolinggo', 60.95, 3.84, 9.78, 19.15, 0.74, 1.23, 1.59, 0.42, 2.3),
(707, 2023, 'Jawa Timur', 'Kota Mojokerto', 73.1, 0.02, 9.51, 13.45, 0.32, 0.9, 0.11, 0.49, 2.1),
(708, 2023, 'Jawa Timur', 'Kota Madiun', 81, 1.5, 7.9, 4, 2.5, 0.9, 0.9, 0.8, 0.5),
(709, 2023, 'Jawa Timur', 'Kota Surabaya', 55.48, 2.25, 3.05, 22.01, 0.25, 5.75, 1.35, 0.5, 9.36),
(710, 2023, 'Banten', 'Kab. Pandeglang', 58.27, 7.43, 6.42, 17.45, 1.54, 1.04, 0.56, 0.81, 6.48),
(711, 2023, 'Banten', 'Kab. Lebak', 30, 12, 10, 15, 8, 5, 5, 10, 5),
(712, 2023, 'Banten', 'Kota Tangerang', 57.65, 0.44, 13.18, 19.66, 1.07, 2.32, 0.8, 0.37, 4.51),
(713, 2023, 'Banten', 'Kota Serang', 67.03, 5.11, 7.26, 20.16, 0.25, 0.03, 0.11, 0.04, 0.01),
(714, 2023, 'Bali', 'Kab. Jembrana', 27.37, 47.94, 0.84, 19.04, 0.71, 0.03, 3.13, 0.61, 0.33),
(715, 2023, 'Bali', 'Kab. Tabanan', 29.21, 39.76, 2.84, 10.51, 0.33, 2.28, 1.94, 0.94, 12.19),
(716, 2023, 'Bali', 'Kab. Klungkung', 23.38, 46.42, 4.93, 11.93, 1.51, 1.41, 0.46, 1.67, 8.29),
(717, 2023, 'Bali', 'Kab. Bangli', 11.08, 68, 2.1, 6.49, 5.6, 0.08, 1.05, 3.6, 2),
(718, 2023, 'Bali', 'Kab. Karangasem', 60, 9, 18, 8, 1, 1, 1, 1, 1),
(719, 2023, 'Bali', 'Kota Denpasar', 26.28, 43.38, 8.26, 11.1, 2.91, 1.04, 3.09, 1.94, 2),
(720, 2023, 'Nusa Tenggara Barat', 'Kab. Sumbawa', 38.21, 8.18, 12.19, 20.34, 2.36, 0.51, 0.68, 4.28, 13.25),
(721, 2023, 'Nusa Tenggara Barat', 'Kota Mataram', 58, 8, 3, 12, 8, 3, 3, 2, 3),
(722, 2023, 'Nusa Tenggara Timur', 'Kab. Kupang', 44, 21, 6, 12, 2, 6, 3, 1, 5),
(723, 2023, 'Nusa Tenggara Timur', 'Kab. Ngada', 10, 10, 20, 25, 10, 10, 5, 5, 5),
(724, 2023, 'Nusa Tenggara Timur', 'Kab. Lembata', 32, 15, 10, 20, 3, 5, 2, 2, 11),
(725, 2023, 'Nusa Tenggara Timur', 'Kab. Rote Ndao', 20, 20, 10, 40, 2, 2, 1, 1, 4),
(726, 2023, 'Nusa Tenggara Timur', 'Kab. Nagekeo', 15.86, 11.46, 22.16, 26.31, 5.29, 6.17, 2.96, 5.56, 4.23),
(727, 2023, 'Nusa Tenggara Timur', 'Kab. Sumba Tengah', 33, 26, 11, 19, 1, 2, 1, 1, 6),
(728, 2023, 'Nusa Tenggara Timur', 'Kab. Malaka', 35.43, 28.93, 10.01, 12.93, 0.54, 2.37, 1.97, 0.39, 7.43),
(729, 2023, 'Nusa Tenggara Timur', 'Kota Kupang', 29, 17, 13, 19.5, 4.25, 3, 2.8, 3.3, 8.15),
(730, 2023, 'Kalimantan Barat', 'Kab. Sambas', 40, 5.6, 14.2, 25, 0.3, 1, 0.9, 2, 11),
(731, 2023, 'Kalimantan Barat', 'Kab. Mempawah', 40, 13, 11, 19, 3, 3, 2, 2, 7),
(732, 2023, 'Kalimantan Barat', 'Kab. Kapuas Hulu', 42.15, 1.32, 12.8, 28.62, 3.2, 1.82, 3.56, 4.29, 2.24),
(733, 2023, 'Kalimantan Barat', 'Kab. Sekadau', 80, 8, 3, 7, 0.3, 0.2, 0.1, 1.1, 0.3),
(734, 2023, 'Kalimantan Barat', 'Kab. Melawi', 38.39, 0.28, 12.99, 46.2, 0.08, 0.38, 0.02, 0.06, 1.6),
(735, 2023, 'Kalimantan Tengah', 'Kab. Kotawaringin Barat', 55, 5, 10, 12, 2, 2, 2, 1, 11),
(736, 2023, 'Kalimantan Tengah', 'Kab. Barito Selatan', 47.36, 9.82, 7, 22.2, 3, 1, 2, 2, 5.62),
(737, 2023, 'Kalimantan Tengah', 'Kab. Barito Utara', 30, 12, 14, 20, 5, 5, 2, 2, 10),
(738, 2023, 'Kalimantan Tengah', 'Kab. Katingan', 56.98, 2.06, 13.17, 14.34, 4.02, 1.78, 3.41, 3.28, 0.95),
(739, 2023, 'Kalimantan Tengah', 'Kab. Seruyan', 30, 15, 10.5, 25, 2, 1.5, 5, 2.5, 8.5),
(740, 2023, 'Kalimantan Tengah', 'Kab. Sukamara', 56.13, 4.12, 4.81, 21.06, 6.88, 5.16, 0.01, 1.09, 0.73),
(741, 2023, 'Kalimantan Tengah', 'Kab. Pulang Pisau', 23.51, 26.74, 13.31, 25.13, 3.19, 2.41, 0.83, 1.79, 3.09),
(742, 2023, 'Kalimantan Tengah', 'Kab. Barito Timur', 32, 9, 7, 37, 1, 5, 6.5, 1.2, 1.3),
(743, 2023, 'Kalimantan Tengah', 'Kota Palangkaraya', 42.05, 4.33, 8.66, 19.49, 2.31, 2.36, 2.18, 2.04, 16.58),
(744, 2023, 'Kalimantan Selatan', 'Kab. Kotabaru', 68, 0.7, 12.5, 12.4, 0.9, 0.4, 0.4, 1.1, 3.6),
(745, 2023, 'Kalimantan Selatan', 'Kab. Barito Kuala', 59.5, 9.5, 10.5, 12, 3.1, 1.9, 0.5, 2, 1),
(746, 2023, 'Kalimantan Selatan', 'Kab. Hulu Sungai Selatan', 55.18, 1.48, 13.13, 25.35, 2.9, 0.46, 0.9, 0.4, 0.2),
(747, 2023, 'Kalimantan Selatan', 'Kab. Tabalong', 36, 2, 17, 24, 1, 3, 1, 3, 13),
(748, 2023, 'Kalimantan Selatan', 'Kab. Tanah Bumbu', 64.04, 4.23, 3.2, 12.3, 2.1, 2.2, 1.24, 2.23, 8.46),
(749, 2023, 'Kalimantan Selatan', 'Kab. Balangan', 60.46, 7.31, 10.66, 15.73, 2.07, 1.64, 0.82, 0.2, 1.11),
(750, 2023, 'Kalimantan Selatan', 'Kota Banjarmasin', 27.89, 7.33, 9.2, 20.61, 3.21, 4.3, 5.26, 4.39, 17.81),
(751, 2023, 'Kalimantan Selatan', 'Kota Banjarbaru', 48.3, 4.2, 16, 23.7, 0.5, 0.3, 0.4, 0.1, 6.5),
(752, 2023, 'Kalimantan Timur', 'Kab. Paser', 54.99, 0.22, 10.2, 13.64, 1.38, 13.56, 1.2, 3.52, 1.29),
(753, 2023, 'Kalimantan Timur', 'Kab. Kutai Kartanegara', 73.92, 0.98, 10.43, 7.86, 2.04, 1.57, 0.55, 1.75, 0.9),
(754, 2023, 'Kalimantan Timur', 'Kab. Berau', 55.39, 1.28, 3.74, 26.47, 9.63, 0.2, 0.2, 0.99, 2.1),
(755, 2023, 'Kalimantan Timur', 'Kab. Kutai Timur', 41.05, 19.71, 11.15, 21.06, 2.97, 1.02, 1, 1.04, 1),
(756, 2023, 'Kalimantan Timur', 'Kab. Penajam Paser Utara', 46.13, 1.09, 20.87, 29.83, 0.22, 0.02, 0.67, 0.83, 0.34),
(757, 2023, 'Kalimantan Timur', 'Kab. Mahakam Ulu', 59, 15, 10, 10, 2, 0.5, 0.5, 2, 1),
(758, 2023, 'Kalimantan Timur', 'Kota Balikpapan', 42.3, 6.36, 10.26, 7.2, 3.87, 2.94, 0.98, 6.56, 19.53),
(759, 2023, 'Kalimantan Timur', 'Kota Samarinda', 52.88, 8.55, 11.41, 12.94, 0.35, 0.9, 0.33, 0.71, 11.93),
(760, 2023, 'Kalimantan Timur', 'Kota Bontang', 44.12, 6, 27.5, 18, 2.1, 0.03, 0.25, 1.05, 0.95),
(761, 2023, 'Kalimantan Utara', 'Kab. Nunukan', 52.38, 1.77, 11.81, 12.57, 3, 4.23, 1.08, 1.68, 11.48),
(762, 2023, 'Sulawesi Utara', 'Kab. Bolaang Mongondow', 49.98, 14.52, 8.5, 10.8, 4.05, 3.33, 2.9, 2.32, 3.6),
(763, 2023, 'Sulawesi Utara', 'Kab. Kepulauan Sangihe', 45.58, 8.95, 5.2, 29.6, 0.46, 0.5, 3.3, 2, 4.41),
(764, 2023, 'Sulawesi Utara', 'Kab. Kepulauan Talaud', 40.58, 9, 5.2, 32.5, 0.3, 0.3, 2, 1.7, 8.42),
(765, 2023, 'Sulawesi Utara', 'Kab. Minahasa Selatan', 20, 12, 12, 36, 4, 4, 5, 2, 5),
(766, 2023, 'Sulawesi Utara', 'Kab. Minahasa Utara', 46.5, 13.65, 12.25, 15.55, 2.2, 2.35, 2.55, 2.5, 2.45),
(767, 2023, 'Sulawesi Utara', 'Kab. Bolaang Mongondow Utara', 20, 25, 10, 10, 10, 5, 5, 5, 10),
(768, 2023, 'Sulawesi Utara', 'Kab. Bolaang Mongondow Selatan', 16, 19, 10, 11, 6, 7, 9, 6, 16),
(769, 2023, 'Sulawesi Utara', 'Kota Manado', 47.7, 1.14, 18.66, 19.48, 0.26, 1.54, 1.49, 1.85, 7.88),
(770, 2023, 'Sulawesi Utara', 'Kota Bitung', 50.2, 2.05, 16.03, 15.07, 0.36, 1.48, 1.37, 1.83, 11.61),
(771, 2023, 'Sulawesi Utara', 'Kota Kotamobagu', 53.5, 5, 11.8, 8.5, 5.7, 3.6, 2.6, 2.8, 6.5),
(772, 2023, 'Sulawesi Tengah', 'Kab. Banggai', 33, 12, 21, 15, 6, 3, 2, 2, 6),
(773, 2023, 'Sulawesi Tengah', 'Kab. Morowali', 2.2, 3, 2.5, 8.88, 3.1, 1.5, 2, 0.2, 3),
(774, 2023, 'Sulawesi Tengah', 'Kab. Banggai Kepulauan', 15, 20, 13, 15, 8, 3, 5, 6, 15),
(775, 2023, 'Sulawesi Tengah', 'Kota Palu', 70.96, 0.77, 9.44, 10.43, 1.9, 1.06, 0.6, 0.59, 4.25),
(776, 2023, 'Sulawesi Selatan', 'Kab. Bantaeng', 30.13, 27.5, 11, 14, 1.24, 3, 2, 1.13, 10),
(777, 2023, 'Sulawesi Selatan', 'Kab. Jeneponto', 19, 22.5, 11.5, 21.5, 5, 1.5, 4, 2.5, 12.5),
(778, 2023, 'Sulawesi Selatan', 'Kab. Takalar', 50, 20, 10, 10, 2, 1, 1, 1, 5),
(779, 2023, 'Sulawesi Selatan', 'Kab. Gowa', 56.55, 1.21, 9.09, 12.68, 2.39, 0.94, 0.47, 1.51, 15.16),
(780, 2023, 'Sulawesi Selatan', 'Kab. Bone', 39.2, 13.3, 11.4, 18.8, 3.4, 2.6, 2.1, 2.1, 7.1),
(781, 2023, 'Sulawesi Selatan', 'Kab. Pangkajene Kepulauan', 63.13, 10.83, 8.99, 7.29, 6.5, 0.72, 0.19, 1.55, 0.8),
(782, 2023, 'Sulawesi Selatan', 'Kab. Barru', 61, 9.5, 5.5, 15.5, 1.7, 1.5, 1, 1.1, 3.2),
(783, 2023, 'Sulawesi Selatan', 'Kab. Soppeng', 53, 7, 14, 20, 2.2, 0.7, 2, 0.3, 0.8),
(784, 2023, 'Sulawesi Selatan', 'Kab. Sidenreng Rappang', 41.4, 13.2, 10.9, 18.5, 3, 2.6, 2, 2, 6.4),
(785, 2023, 'Sulawesi Selatan', 'Kab. Luwu Utara', 65, 10, 9, 2, 1, 1, 1, 1, 10),
(786, 2023, 'Sulawesi Selatan', 'Kab. Luwu Timur', 40, 10, 25, 15, 3, 1.5, 2, 2, 1.5),
(787, 2023, 'Sulawesi Selatan', 'Kota Parepare', 51.4, 11.6, 11.9, 12.25, 2.85, 2.3, 2.15, 2.95, 2.6),
(788, 2023, 'Sulawesi Tenggara', 'Kab. Kolaka', 65, 5, 10, 10, 5, 1, 1, 2, 1),
(789, 2023, 'Sulawesi Tenggara', 'Kab. Muna', 47, 9, 12, 23, 2, 2, 2, 2, 1),
(790, 2023, 'Sulawesi Tenggara', 'Kab. Buton', 10, 10, 2, 50, 5, 2, 3, 1, 17),
(791, 2023, 'Sulawesi Tenggara', 'Kab. Wakatobi', 43, 8, 9, 27, 4, 3, 1, 2, 3),
(792, 2023, 'Sulawesi Tenggara', 'Kab. Kolaka Timur', 30, 15, 9, 25, 1, 5, 10, 2, 3),
(793, 2023, 'Sulawesi Tenggara', 'Kab. Buton Selatan', 65.83, 3.36, 4.52, 12.48, 1.44, 3.79, 1.95, 1.89, 4.74),
(794, 2023, 'Sulawesi Tenggara', 'Kota Kendari', 43.65, 17.44, 16.78, 17.25, 1.54, 0.72, 1.06, 1.16, 0.4),
(795, 2023, 'Maluku', 'Kab. Kepulauan Tanimbar', 20, 65, 50, 100, 5, 60, 85, 25, 30),
(796, 2023, 'Maluku', 'Kota Ambon', 39, 9, 12, 16, 7, 4, 3, 5, 5),
(797, 2023, 'Papua', 'Kab. Biak Numfor', 13, 8, 7.5, 15, 25, 6.5, 9.5, 10, 5.5),
(798, 2023, 'Papua', 'Kota Jayapura', 56, 10.5, 4.9, 11.79, 4.29, 3.04, 2.09, 1.45, 5.94),
(799, 2023, 'Papua Barat', 'Kab. Kaimana', 15, 15, 15, 25, 10, 5, 5, 5, 5),
(800, 2023, 'Papua Selatan', 'Kab. Boven Digoel', 5, 5, 25, 25, 5, 5, 10, 5, 15),
(801, 2023, 'Papua Selatan', 'Kab. Asmat', 20, 11.8, 15, 40, 4, 2.4, 3, 1.8, 2),
(802, 2023, 'Papua Pegunungan', 'Kab. Jayawijaya', 22, 10, 10, 25, 7, 10, 3, 10, 3),
(803, 2023, 'Papua Barat Daya', 'Kab. Raja Ampat', 10, 20, 20, 25, 8, 7, 3, 2, 5),
(804, 2023, 'Papua Barat Daya', 'Kota Sorong', 15, 25, 10, 30, 5, 5, 1, 5, 4);

-- --------------------------------------------------------

--
-- Struktur dari tabel `konten`
--

CREATE TABLE `konten` (
  `id` int(11) NOT NULL,
  `judul` varchar(255) DEFAULT NULL,
  `deskripsi` text DEFAULT NULL,
  `tanggal` date DEFAULT NULL,
  `gambar` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data untuk tabel `konten`
--

INSERT INTO `konten` (`id`, `judul`, `deskripsi`, `tanggal`, `gambar`) VALUES
(34, 'Pentingnya Pengelolaan Sampah untuk Lingkungan yang Lebih Bersih', 'Pengelolaan sampah yang efektif dan efisien merupakan langkah penting untuk menjaga kebersihan lingkungan dan mengurangi dampak negatif terhadap kesehatan serta ekosistem. Melalui pemilahan sampah, daur ulang, dan pengelolaan sampah yang ramah lingkungan, kita dapat mengurangi volume sampah yang masuk ke tempat pembuangan akhir (TPA), mengurangi polusi, dan menciptakan peluang ekonomi baru melalui daur ulang dan pengolahan sampah menjadi energi. Pengelolaan sampah yang baik juga berperan dalam meningkatkan kesadaran masyarakat tentang pentingnya kebersihan dan keberlanjutan.', '2024-12-27', 'images/lingkungan-bersih.jpg'),
(35, 'Pengelolaan Sampah yang Efektif untuk Lingkungan yang Sehat', 'Pengelolaan sampah yang baik adalah kunci untuk menjaga kebersihan lingkungan dan mencegah pencemaran. Proses ini mencakup pemilahan sampah, daur ulang, dan pengolahan sampah menjadi produk yang berguna, seperti energi atau bahan baku baru. Dengan mengelola sampah secara efektif, kita dapat mengurangi volume sampah yang masuk ke tempat pembuangan akhir (TPA), mengurangi emisi gas rumah kaca, dan menciptakan peluang ekonomi baru melalui industri daur ulang. Pengelolaan sampah yang berkelanjutan juga penting untuk meningkatkan kesadaran masyarakat tentang pentingnya kebersihan dan keberlanjutan.', '2024-12-27', 'images/pengelolaan-sampah.jpg'),
(36, 'Optimalisasi Pengelolaan Sampah untuk Mewujudkan Kota Ramah Lingkungan', 'Pengelolaan sampah yang efektif berperan penting dalam menciptakan lingkungan yang sehat dan berkelanjutan. Dengan memanfaatkan teknologi terbaru dan pendekatan yang berbasis data, kita dapat meminimalkan produksi sampah dan memaksimalkan potensi daur ulang. Sistem pengelolaan sampah yang cerdas mencakup pemilahan sampah, pengurangan volume sampah yang masuk ke tempat pembuangan akhir (TPA), serta pemanfaatan kembali sampah untuk sumber daya energi dan ekonomi. Implementasi pengelolaan sampah yang optimal tidak hanya mendukung keberlanjutan lingkungan, tetapi juga meningkatkan kualitas hidup masyarakat dengan menciptakan ruang yang lebih bersih dan nyaman.', '2025-01-18', 'images/Foto-1-Kota-Deltamas-1024x769.jpeg');

-- --------------------------------------------------------

--
-- Struktur dari tabel `users`
--

CREATE TABLE `users` (
  `id_users` int(11) NOT NULL,
  `username` varchar(100) NOT NULL,
  `password` varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data untuk tabel `users`
--

INSERT INTO `users` (`id_users`, `username`, `password`) VALUES
(1, 'adminpengelola', 'admin!!!');

--
-- Indexes for dumped tables
--

--
-- Indeks untuk tabel `datasetbersih`
--
ALTER TABLE `datasetbersih`
  ADD PRIMARY KEY (`id_wilayah`);

--
-- Indeks untuk tabel `konten`
--
ALTER TABLE `konten`
  ADD PRIMARY KEY (`id`);

--
-- Indeks untuk tabel `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id_users`);

--
-- AUTO_INCREMENT untuk tabel yang dibuang
--

--
-- AUTO_INCREMENT untuk tabel `datasetbersih`
--
ALTER TABLE `datasetbersih`
  MODIFY `id_wilayah` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=805;

--
-- AUTO_INCREMENT untuk tabel `konten`
--
ALTER TABLE `konten`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=37;

--
-- AUTO_INCREMENT untuk tabel `users`
--
ALTER TABLE `users`
  MODIFY `id_users` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
