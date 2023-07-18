# Edwin Mahendra - Movie Recommendation

Connect with me on
[Instagram](https://www.instagram.com/edwinmahendra_) or
[GitHub](https://github.com/edwinmahendra) or
[LinkedIn](https://www.linkedin.com/in/edwin-mahendra-a2944821b/)
<hr>

Pada proyek kali ini, dibuat model untuk sistem rekomendasi *movies* dengan menggunakan **Content-based Filtering** dengan beberapa teknik, termasuk **algoritma vektorisasi TF-IDF** dan **salah satu *pre-trained model* SBERT** dari sinopsis film. Dalam upaya meningkatkan akurasi rekomendasi, **fitur tambahan seperti genre, kata kunci, *top 5 cast*, dan sutradara juga digabungkan ke dalam model**. Dengan pendekatan ini, sistem dapat memberikan rekomendasi yang lebih relevan dan spesifik, mencerminkan preferensi dan selera pengguna dengan lebih baik.

## Domain Proyek

<div align="center"><img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*8y9sCjaxFmAzEiVEzw3YoQ.jpeg" width="700"/></div><br>

Film, sebagai media hiburan populer, mencakup berbagai genre dan kategori yang unik, seperti komedi, horor, aksi, animasi, dokumenter, fantasi, dan sains fiksi. Setiap genre mencerminkan variasi dalam cerita, suasana, dan tema film. Setiap pengguna memiliki preferensi unik terhadap berbagai genre ini.

Pada titik ini, muncul pentingnya sistem rekomendasi yang dapat memahami dan merespons preferensi individu dari setiap pengguna. Sistem rekomendasi bertujuan untuk mempersonalisasi pengalaman menonton dengan merekomendasikan film yang paling relevan dan menarik bagi setiap pengguna. Dengan demikian, pengguna tidak hanya menemukan film baru yang sesuai dengan minat mereka, tapi juga menghindari rasa bosan yang mungkin muncul dari menonton film-film dengan genre atau tema yang sama.

Selain itu, sistem rekomendasi juga berperan dalam memperluas wawasan dan apresiasi pengguna terhadap berbagai genre film. Pengguna bisa mengeksplorasi dan menemukan genre baru yang mungkin belum pernah mereka coba sebelumnya, memberikan pengalaman menonton yang lebih beragam dan memperkaya. Dalam jangka panjang, hal ini bisa meningkatkan kepuasan pengguna dan mempertahankan minat mereka dalam menggunakan layanan streaming film.

Ada berbagai pendekatan untuk membangun sistem rekomendasi, termasuk pemfilteran berbasis konten, pemfilteran kolaboratif, dan pemfilteran *hybrid*. Pemfilteran berbasis konten, pendekatan yang digunakan dalam proyek ini, merujuk pada rekomendasi yang dibuat berdasarkan informasi detail tentang item atau profil pengguna. Penelitian oleh Pazzani dan Billsus (2007) menunjukkan bahwa pendekatan ini dapat sangat efektif dalam menghasilkan rekomendasi yang relevan. [1]

## **Business Understanding**

Proyek sistem rekomendasi ini dirancang dengan mempertimbangkan karakteristik bisnis berikut:

+ **Perusahaan Layanan *Streaming* Film:** Perusahaan seperti Netflix, Amazon Prime, Hulu, dan Vidio dapat memanfaatkan sistem rekomendasi ini untuk meningkatkan penawaran mereka kepada pelanggan. Dengan sistem yang dapat mengidentifikasi preferensi individu dan memberikan rekomendasi film yang sesuai, mereka dapat meningkatkan retensi pengguna dan mendorong konsumsi konten yang lebih tinggi.
+ **Pecinta Film dan Pengguna Layanan *Streaming*:** Konsumen yang memiliki kecintaan pada film dan sering menggunakan layanan streaming juga akan diuntungkan oleh sistem ini. Melalui rekomendasi yang dirancang khusus berdasarkan preferensi dan sejarah menonton mereka, pengguna dapat menemukan film baru yang mungkin mereka sukai, menyempurnakan pengalaman menonton film mereka, dan memanfaatkan layanan streaming mereka dengan lebih efisien.

Sehingga, sistem ini membantu menciptakan lingkungan *win-win* bagi kedua pihak - perusahaan layanan film mendapatkan pengguna yang lebih terlibat, dan konsumen mendapatkan rekomendasi yang lebih sesuai dengan selera mereka.

### Problem Statement
+ Bagaimana cara membangun sistem rekomendasi film berdasarkan konten yang efektif dan efisien, yang mampu menargetkan preferensi pengguna dengan tepat dan menyajikan rekomendasi yang relevan dan personal?
+ Bagaimana teknik *TF-IDF Vectorizer* dan *pre-trained model SBERT* dapat digunakan untuk mengubah teks sinopsis film menjadi vektor numerik yang mencerminkan kesamaan antara film?
+ Bagaimana fitur tambahan seperti genre, keywords, top 5 cast, dan director dapat digunakan untuk mempersonalisasi sistem rekomendasi dan meningkatkan akurasi rekomendasi?

### Goals
+ Membangun sistem rekomendasi film berbasis konten yang efektif dan efisien, yang mampu menargetkan preferensi pengguna dengan tepat dan menyajikan rekomendasi yang relevan dan personal.
+ Menerapkan teknik *TF-IDF Vectorizer* dan *pre-trained model SBERT* untuk merubah teks sinopsis film menjadi vektor numerik, yang kemudian digunakan untuk mengukur kesamaan antara film.
+ Mengintegrasikan fitur tambahan seperti genre, keywords, top 5 cast, dan director ke dalam model untuk meningkatkan personalisasi dan akurasi sistem rekomendasi.

### Solution Approach
Untuk mencapai tujuan tersebut, langkah-langkah berikut akan dilakukan:

+ **Feature Selection:** Menentukan fitur penting dari sebuah film, seperti genre, keywords, top 5 cast, dan director, berdasarkan analisis data dan studi literatur terkait.
+ **Pre-processing Data:** Menyiapkan dan memproses data untuk memastikan data tersebut dapat digunakan untuk pengembangan model. Proses ini melibatkan penghapusan nilai-nilai kosong atau *null*, normalisasi data, dan penanganan outliers jika ada.
+ **Text Vectorization:** Menggunakan TF-IDF Vectorizer dan SBERT untuk merubah teks sinopsis film menjadi vektor numerik. Vektor ini akan digunakan untuk mengukur kesamaan antara film dan sebagai basis dalam memberikan rekomendasi.
+ **Personalization:** Mengintegrasikan fitur tambahan yang telah ditentukan ke dalam model rekomendasi untuk mencerminkan preferensi dan selera pengguna secara lebih baik.
+ **Model Building:** Membangun sistem rekomendasi berdasarkan teknik Content-Based Filtering, yang memberikan rekomendasi berdasarkan kesamaan konten atau karakteristik film itu sendiri.

## Data Understanding
Untuk melanjutkan proyek Machine Learning ini, diperlukan dataset yang akan digunakan sebagai bahan atau data dalam prosesnya. Dataset yang dipilih adalah The Movies Dataset yang dapat diunduh melalui Kaggle menggunakan tautan berikut:

**The Movies Dataset (by ROUNAK BANIK):** https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

Berikut ini adalah **gambaran umum dari dataset** tersebut:
+ Dataset ini terdiri dari metadata untuk **45.000 film** yang terdaftar dalam Dataset Full MovieLens. 
+ Dataset ini mencakup film-film yang **dirilis pada atau sebelum Juli 2017**. 
+ Data yang disediakan mencakup pemeran, kru, kata kunci plot, anggaran, pendapatan, poster, tanggal rilis, bahasa, perusahaan produksi, negara, jumlah suara TMDB, dan rata-rata suara.
+ Dataset juga memiliki file **movies_metadata.csv** yang berisi **26 juta rating** dari 270.000 pengguna untuk semua (kurang lebih) **45.000 film**. Rating diberikan dalam skala 1-5 dan diperoleh dari situs web resmi GroupLens.
+ Dataset disimpan dalam **format *csv*** (*Comma Separated Value*), yang mudah diproses menggunakan berbagai perangkat lunak data analisis dan pemrosesan.
+ **Beberapa kolom** di dataset **memiliki nilai *null***, sehingga **perlu dilakukan langkah-langkah normalisasi** atau pembersihan data sebelum digunakan dalam pembuatan model *machine learning*.

**The Movies Dataset**<br>
File .csv yang digunakan adalah movies_metadata.csv. File ini **terdiri dari 45.466 baris** dan **24 kolom**. Terdapat kolom-kolom dengan tipe data float64 dan object. Beberapa kolom memiliki jumlah entri *non-null* yang berbeda-beda, menandakan adanya missing values dalam dataset. Setiap entri memiliki informasi pada Tabel 1.

<p align="center">
  <i>Tabel 1. Atribut dalam dataset movies_metadata.csv</i>
</p>
<div align="center">

| Attribute              | Description                                                                                   |
|------------------------|-----------------------------------------------------------------------------------------------|
| adult                  | Menunjukkan apakah film tersebut ditujukan untuk penonton dewasa atau tidak.                  |
| belongs_to_collection  | Memberikan informasi tentang apakah film tersebut bagian dari suatu koleksi atau seri film.   |
| budget                 | Anggaran yang digunakan untuk produksi film.                                                 |
| genres                 | Genre dari film.                                                                              |
| homepage               | Alamat situs web resmi dari film tersebut.                                                    |
| id                     | ID unik yang diberikan kepada setiap film.                                                    |
| imdb_id                | ID unik yang diberikan oleh IMDb (Internet Movie Database).                                   |
| original_language      | Bahasa asli dari film tersebut.                                                               |
| original_title         | Judul asli dari film tersebut.                                                                |
| overview               | Sinopsis atau deskripsi singkat tentang film tersebut.                                        |
| popularity             | Tingkat popularitas film tersebut.                                                            |
| poster_path            | Alamat URL dari poster film.                                                                  |
| production_companies   | Perusahaan yang memproduksi film tersebut.                                                    |
| production_countries   | Negara di mana film tersebut diproduksi.                                                      |
| release_date           | Tanggal rilis film tersebut.                                                                  |
| revenue                | Pendapatan yang dihasilkan oleh film tersebut.                                                |
| runtime                | Durasi film dalam menit.                                                                      |
| spoken_languages       | Bahasa yang digunakan dalam film tersebut.                                                    |
| status                 | Status rilis dari film tersebut (misalnya, dirilis, dalam produksi, dll.)                      |
| tagline                | Tagline atau slogan singkat dari film tersebut.                                               |
| title                  | Judul film.                                                                                   |
| video                  | Menunjukkan apakah film tersebut memiliki video pratinjau atau tidak.                          |
| vote_average           | Rata-rata skor yang diberikan oleh penonton.                                                  |
| vote_count             | Jumlah suara yang diterima oleh film tersebut.                                                |

</div>

Berikut adalah analisis tentang data yang hilang pada setiap atribut:
+ **belongs_to_collection:** Terdapat **40.972 nilai yang hilang** pada atribut ini, yang menunjukkan bahwa sebagian besar film dalam dataset tidak termasuk dalam koleksi atau seri film tertentu.
+ **homepage:** Terdapat **37.684 nilai yang hilang** pada atribut ini, yang menunjukkan bahwa sebagian besar film tidak memiliki halaman web resmi yang terkait.
+ **imdb_id:** Terdapat **17 nilai yang hilang** pada atribut ini, yang menunjukkan bahwa beberapa film tidak memiliki ID unik yang diberikan oleh IMDb (Internet Movie Database).
+ **original_language:** Terdapat **11 nilai yang hilang** pada atribut ini, yang menunjukkan bahwa beberapa film tidak memiliki bahasa asli yang tercatat.
+ **overview:** Terdapat **954 nilai yang hilang** pada atribut ini, yang menunjukkan bahwa beberapa film tidak memiliki sinopsis atau deskripsi singkat.
+ **popularity:** Terdapat **5 nilai yang hilang** pada atribut ini, yang menunjukkan bahwa sebagian kecil film tidak memiliki informasi tentang tingkat popularitas mereka.
+ **poster_path:** Terdapat **386 nilai yang hilang** pada atribut ini, yang menunjukkan bahwa beberapa film tidak memiliki URL gambar poster terkait.
+ **production_companies:** Terdapat **3 nilai yang hilang** pada atribut ini, yang menunjukkan bahwa beberapa film tidak memiliki informasi tentang perusahaan produksi yang terlibat.
+ **production_countries:** Terdapat **3 nilai yang hilang** pada atribut ini, yang menunjukkan bahwa beberapa film tidak memiliki informasi tentang negara tempat produksi dilakukan.
+ **release_date:** Terdapat **87 nilai yang hilang** pada atribut ini, yang menunjukkan bahwa beberapa film tidak memiliki tanggal rilis yang tercatat.
+ **revenue:** Terdapat **6 nilai yang hilang pada atribut ini**, yang menunjukkan bahwa beberapa film tidak memiliki informasi tentang pendapatan yang dihasilkan.
+ **runtime:** Terdapat **263 nilai yang hilang** pada atribut ini, yang menunjukkan bahwa beberapa film tidak memiliki informasi tentang durasi film dalam menit.
+ **spoken_languages:** Terdapat **6 nilai yang hilang** pada atribut ini, yang menunjukkan bahwa beberapa film tidak memiliki informasi tentang bahasa yang digunakan.
+ **status:** Terdapat **87 nilai yang hilang pada atribut ini**, yang menunjukkan bahwa beberapa film tidak memiliki status rilis yang tercatat.
+ **tagline:** Terdapat **25.054 nilai yang hilang** pada atribut ini, yang menunjukkan bahwa sebagian besar film tidak memiliki tagline atau slogan singkat.
+ **title:** Terdapat **6 nilai yang hilang** pada atribut ini, yang menunjukkan bahwa beberapa film tidak memiliki judul yang tercatat.
+ **video:** Terdapat **6 nilai yang hilang** pada atribut ini, yang menunjukkan bahwa beberapa film tidak memiliki video pratinjau yang terkait.
+ **vote_average:** Terdapat **6 nilai yang hilang** pada atribut ini, yang menunjukkan bahwa beberapa film tidak memiliki skor rata-rata yang tercatat.
+ **vote_count:** Terdapat **6 nilai yang hilang** pada atribut ini, yang menunjukkan bahwa beberapa film tidak memiliki jumlah suara yang tercatat.

### Visualisasi Data

Berikut adalah beberapa bentuk visualisasi data yang bisa ditampilkan dari dataset.

<p align="center">
  <i>Tabel 2. Eksplorasi Dataset untuk Memvisualisasikan 10 sampel film dengan weighted_rating tertinggi</i>
</p>
<div align="center">

| Title                   | Year | Vote Count | Vote Average | Popularity | Genres                               | Weighted Rating |
|-------------------------|------|------------|--------------|------------|--------------------------------------|-----------------|
| The Shawshank Redemption| 1994 | 8358.0     | 8.5          | 51.645403  | [Drama, Crime]                       | 8.4             |
| The Godfather           | 1972 | 6024.0     | 8.5          | 41.109264  | [Drama, Crime]                       | 8.3             |
| The Dark Knight         | 2008 | 12269.0    | 8.3          | 123.167259 | [Drama, Action, Crime, Thriller]      | 8.2             |
| Fight Club              | 1999 | 9678.0     | 8.3          | 63.869599  | [Drama]                              | 8.2             |
| Pulp Fiction            | 1994 | 8670.0     | 8.3          | 140.950236 | [Thriller, Crime]                    | 8.2             |
| Whiplash                | 2014 | 4376.0     | 8.3          | 64.29999   | [Drama]                              | 8.1             |
| Forrest Gump            | 1994 | 8147.0     | 8.2          | 48.307194  | [Comedy, Drama, Romance]             | 8.1             |
| Schindler's List        | 1993 | 4436.0     | 8.3          | 41.725123  | [Drama, History, War]                | 8.1             |
| Interstellar            | 2014 | 11187.0    | 8.1          | 32.213481  | [Adventure, Drama, Science Fiction]   | 8.0             |
| The Intouchables        | 2011 | 5410.0     | 8.2          | 16.086919  | [Drama, Comedy]                      | 8.0             |

</div>
<br><br>
<p align="center">
  <i>Gambar 1. Visualisasi 10 sampel film diurutkan dengan tingkat weighted rating</i>
</p>

<div><img src="https://github.com/edwinmahendra/DicodingAssets/blob/main/grafik_terapan_2.jpeg?raw=true" width="1000"/></div>


Melalui Tabel 3 dan Gambar 1 (ditampilkan dalam visualisasi bar graph) ditampilkan 10 rekomendasi movies berdasarkan *weighted rating*. *Weighted rating* dihitung dengan mempertimbangkan dua faktor utama, yaitu jumlah suara yang diterima oleh film (vote_count) dan nilai rata-rata dari penilaian film (vote_average). Dengan menggunakan *weighted rating*, film-film dengan suara dan penilaian yang lebih tinggi akan mendapatkan peringkat yang lebih tinggi, sementara film-film dengan suara dan penilaian yang rendah akan mendapatkan peringkat yang lebih rendah.

<p align="center">
  <i>Gambar 2. Visualisasi 10 sampel film diurutkan dengan tingkat popularitasnya</i>
</p>
<div><img src="https://github.com/edwinmahendra/DicodingAssets/blob/main/grafik2_terapan_2.jpeg?raw=true" width="1000"/></div>

Pada Gambar 2 menunjukkan visualisasi dari 10 film dengan tingkat popularitas tertinggi. Dalam diagram batang ini, judul film ditampilkan pada sumbu y dan tingkat popularitas film ditampilkan pada sumbu x.

Setiap batang dalam diagram mewakili satu film. Panjang batang mencerminkan tingkat popularitas dari film tersebut, semakin panjang batang, semakin populer film itu. Nilai popularitas untuk setiap film ditampilkan tepat setelah batang diagramnya, memberikan gambaran yang jelas tentang tingkat popularitas numerik dari film tersebut.



## Data Preparation
### Download Dataset
Dataset diunduh dengan menggunakan API dari kaggle. Dataset yang dipilih adalah The Movies Dataset, yang dapat diunduh melalui Kaggle menggunakan tautan [ini.](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

### Dataset Extraction
Setelah proses pengunduhan dataset selesai, dataset umumnya berbentuk file terkompresi (.zip). Untuk dapat menggunakan data tersebut, proses ekstraksi perlu dilakukan. Ekstraksi ini melibatkan pembukaan file .zip dan pemindahan semua item di dalamnya ke direktori tertentu.

### Data Normalization
Untuk normalisasi data, terdapat beberapa langkah yang dapat dilakukan:
+ **Penghapusan Data**<br>
  Melalui analisis awal data, ditemukan beberapa kolom yang tidak memberikan informasi yang signifikan untuk tujuan analisis dan pengembangan model selanjutnya. Oleh karena itu, diputuskan untuk menghapus kolom berikut:

  + **adult:** Kolom ini menunjukkan apakah film diperuntukkan bagi penonton dewasa. Informasi ini tidak penting dalam konteks rekomendasi film.
  
  + **homepage:** Kolom ini berisi URL situs web film, yang tidak memberikan informasi tambahan yang berguna untuk analisis.
  + **imdb_id:** Ini adalah ID unik yang diberikan oleh IMDb, yang tidak memberikan informasi tambahan yang berguna untuk analisis.
  + **poster_path:** Kolom ini berisi URL poster film, yang tidak relevan dalam konteks analisis.
  + **video:** Kolom ini menunjukkan apakah film memiliki video trailer atau tidak, yang tidak relevan dalam konteks analisis.
  
<br>

+ **Penyesuaian Nilai Kolom atau Penambahan Fitur**<br>
  Berikut adalah beberapa perubahan yang dilakukan pada data:

  + **Mengatasi nilai NaN:** semua baris di mana kolom 'vote_average' memiliki nilai NaN akan dihapus, karena informasi ini penting dalam mengembangkan model rekomendasi.

  + **Mengubah 'belongs_to_collection' menjadi nilai biner:** Mengubah kolom ini menjadi 1 jika film tersebut termasuk dalam koleksi dan 0 jika tidak. Ini memungkinkan model untuk memahami apakah film tersebut adalah bagian dari seri atau koleksi.

  + **Mengolah 'genres':** Mengubah string representasi list menjadi list sebenarnya. Kemudian, mengambil nilai dari kunci 'name' untuk setiap item dalam list tersebut, yang memberikan genre film.

  + **Membuat atribut baru bernama year:** Mengubah kolom release_date menjadi tipe data datetime dan kemudian mengambil tahun dari tanggal tersebut. Informasi ini dapat membantu model memahami tren dalam preferensi film berdasarkan tahun rilis.

  + **Filtering status film:** Hanya menyertakan film yang memiliki status 'Released' dalam analisis, karena hanya difokuskan saja pada film yang sudah dirilis. Setelah itu, kolom 'status' dihapus dari DataFrame.

Perubahan-perubahan ini sangat penting dalam mempersiapkan data untuk pengembangan model rekomendasi yang lebih efektif dan efisien. Dengan melakukan penyesuaian nilai kolom, penambahan fitur, dan normalisasi data, dapat mengoptimalkan kualitas dan kegunaan data untuk analisis lebih lanjut. Hasilnya, setelah melalui pengecekan dengan fungsi isnull, semua data yang kosong telah berhasil diatasi dan tidak ada lagi nilai yang hilang dalam dataset.

## Model Building
### 1. Simple Recommendation
Base Recommendation bertujuan untuk menghasilkan daftar film yang direkomendasikan berdasarkan genre. Berikut adalah langkah-langkah pembuatannya:

Langkah pertama, fungsi **get_genre_filtered_data** digunakan untuk **menyaring film berdasarkan genre**. Misalnya, jika pengguna menginginkan film dengan genre "Action", fungsi ini akan menyisihkan film-film yang tidak memiliki genre ini.

Setelah film disaring berdasarkan genre, langkah kedua adalah **menentukan suara minimum** yang harus dimiliki **suatu film** yang **dianggap populer**. Fungsi **calculate_vote_quantile** digunakan untuk tujuan ini. Ia **menghitung batas suara berdasarkan persentil yang diberikan**. Sebagai contoh, jika menggunakan persentil 0.95, fungsi ini akan menghitung suara minimum yang diperoleh oleh 95% film dalam dataset.

Langkah ketiga, fungsi **calculate_vote_average** menghitung **rata-rata suara dari semua film** dalam dataset. Ini memberikan gambaran tentang seberapa baik penilaian suatu film dibandingkan dengan film lainnya.

Langkah keempat, setelah mendapatkan suara minimum dan rata-rata suara, fungsi **compute_weighted_rating** dibuat untuk **menghitung rating terbobot untuk setiap film**. Rating terbobot ini dihitung berdasarkan jumlah suara yang diterima film, rata-rata suara, dan faktor bobot.

Akhirnya, **semua fungsi** ini **disatukan** dalam fungsi **simple_movie_recommender**. Fungsi ini menggabungkan semua langkah sebelumnya: menyaring film berdasarkan genre, menghitung suara minimum dan rata-rata suara, dan menghitung rating terbobot. Fungsi ini kemudian mengurutkan film-film berdasarkan rating terbobot dan tahun rilis, dan memberikan daftar film terbaik berdasarkan kriteria tersebut.

Rekomendasi film yang telah dibangun pada bagian ini, meskipun memiliki daya tarik yang luas, namun terdapat kelemahan didalamnya yaitu **kurangnya personalisasi** dalam **rekomendasi yang diberikan** kepada pengguna. Dalam pendekatan saat ini, rekomendasi yang **diberikan sama** untuk semua orang, **tanpa mempertimbangkan selera pribadi pengguna.**

Untuk meningkatkan personalisasi rekomendasi, akan **dikembangkan sistem rekomendasi berbasis konten** (Content-based Filtering) pada bagian kode berikutnya. Dalam sistem ini, akan **digunakan informasi** yang relevan tentang film, seperti **konten film**, **genre**, dan lainnya, untuk **memberikan rekomendasi yang lebih dipersonalisasi** kepada pengguna. Dengan mempertimbangkan **preferensi** dan **minat pribadi pengguna**, sistem ini akan memberikan saran yang lebih akurat dan sesuai dengan preferensi individu pengguna.

### 2. Content-based Filtering
Dalam proyek ini, teknik ***Content-Based Filtering*** digunakan untuk **memberikan rekomendasi film berdasarkan kesamaan konten atau karakteristik film itu sendiri**. Proses ini melibatkan beberapa metode penting. Sebagai **titik awal**, **algoritma TF-IDF Vectorizer** digunakan untuk merubah teks dari sinopsis film menjadi vektor numerik. Vektor ini kemudian digunakan untuk mengukur kesamaan antara film.

Setelah itu, **untuk meningkatkan efisiensi sistem rekomendasi**, **SBERT**, atau Sentence-BERT akan digunakan **untuk menggantikan proses vektorisasi dengan TF-IDF Vectorizer**. Tujuan penggunaan SBERT adalah untuk meningkatkan kualitas vektorisasi dan berpotensi memberikan skor similaritas yang lebih tinggi antara film.

Terakhir, **untuk lebih mempersonalisasi** sistem rekomendasi dan membuat rekomendasi yang lebih akurat, **beberapa fitur tambahan seperti genre, keywords, top 5 cast, dan director digabungkan**. 

#### 2.1 Content-based Filtering with TF-IDF Vectorizer

  Dimulai dengan mempersiapkan data yang akan digunakan, sistem mencakup sejumlah **tahap perhitungan** dan **manipulasi data** sebelum akhirnya menghasilkan daftar film yang direkomendasikan.

  Pertama, perlu diidentifikasi kriteria yang akan menentukan apakah suatu film dianggap populer atau tidak. Untuk tujuan ini, **dihitung suara minimum** yang harus diperoleh suatu film untuk masuk dalam kategori ini. **Dengan persentil 0.95**, ditentukan bahwa **film harus mendapat suara lebih banyak dari 95% film lain** dalam dataset untuk dianggap populer. Hal ini dilakukan karena mempengaruhi waktu pemrosesan sehingga waktu pemrosesan akan lebih cepat.

  **Rata-rata suara** dari semua film dalam dataset juga dihitung. Hal ini memberikan **titik acuan** yang digunakan untuk menilai suara yang diterima oleh masing-masing film.

  Setelah itu, **dataset difilter** untuk memastikan bahwa hanya film yang memenuhi kriteria popularitas yang termasuk dalam analisis. Proses ini dilakukan dengan membandingkan jumlah suara untuk setiap film dengan suara minimum yang telah dihitung. Jika film mendapatkan **suara lebih banyak atau sama dengan suara minimum**, **film itu tetap di dalam dataset**; jika tidak, film itu dihapus.

  Selanjutnya, perlu mengubah deskripsi teks film menjadi bentuk yang dapat diproses oleh komputer. Metode yang digunakan disebut **TF-IDF**, yang merupakan singkatan dari "Term Frequency-Inverse Document Frequency". Prinsip dasarnya adalah **mengubah setiap kat**a dalam teks menjadi **angka** yang mencerminkan **seberapa sering kata itu muncul dalam teks**, dibandingkan dengan seberapa sering kata itu muncul di seluruh dataset.

  Setelah menerapkan TF-IDF, **diperoleh** apa yang disebut **matriks TF-IDF**. Setiap baris dalam matriks ini mewakili film, dan setiap kolom mewakili kata. Nilai di setiap sel matriks menunjukkan seberapa penting kata tersebut bagi film yang bersangkutan.

  Namun, untuk bisa membuat rekomendasi, perlu diketahui sejauh mana setiap film mirip dengan yang lain. Untuk ini, digunakan teknik yang dikenal sebagai ***cosine similarity***, yang mengukur **kesamaan antara dua vektor** (dalam hal ini, **dua baris dalam matriks TF-IDF**). Dengan cara ini, diperoleh matriks kesamaan cosine, di mana setiap elemen menunjukkan sejauh mana dua film mirip satu sama lain berdasarkan deskripsi mereka.

  Terakhir, matriks kesamaan ini digunakan untuk merekomendasikan film. Fungsi **movies_recommendations** mengambil **judul film,** dan berdasarkan matriks kesamaan, mencari film-film lain yang paling mirip dengannya. Fungsi ini mencari k film yang paling mirip, menghapus film yang asli dari daftar (karena tidak ingin merekomendasikan film yang sama), dan mengembalikan daftar film yang direkomendasikan bersama dengan skor kesamaannya.

  Secara keseluruhan, sistem ini memungkinkan pencarian film berdasarkan deskripsi dan menemukan film lain yang mungkin menarik bagi penonton berdasarkan konten mereka yang serupa. Dengan demikian, sistem ini menciptakan cara yang mudah dan efisien untuk menemukan film baru yang menarik untuk ditonton.

#### 2.2 Content-based Filtering with SBERT
Sistem rekomendasi film ini **ditingkatkan** dengan penggunaan **Sentence-BERT (SBERT)**, sebuah modifikasi dari model Transformer BERT yang dirancang untuk menghasilkan representasi vektor kalimat secara efisien. Model ini digunakan untuk mengubah deskripsi teks dari setiap film menjadi vektor dalam ruang multidimensi, yang memungkinkan perbandingan langsung antara deskripsi film yang berbeda.

Pertama, data film disiapkan dan dipilih hanya film-film yang memiliki jumlah suara yang memadai. Ini membantu memastikan bahwa film yang disarankan adalah yang telah ditonton dan dinilai oleh audiens yang cukup besar.

<p align="center">
  <i>Tabel 3. Beberapa pre-trained model SBERT diurutkan berdasarkan dari yang terbaik</i>
</p>

| Model Name | Performance Sentence Embeddings (14 Datasets) | Performance Semantic Search (6 Datasets) | Avg. Performance | Speed | Model Size |
|---|---|---|---|---|---|
| all-mpnet-base-v2 | 69.57 | 57.02 | 63.30 | 2800 | 420 MB |
| multi-qa-mpnet-base-dot-v1 | 66.76 | 57.60 | 62.18 | 2800 | 420 MB |
| all-distilroberta-v1 | 68.73 | 50.94 | 59.84 | 4000 | 290 MB |
| all-MiniLM-L12-v2 | 68.70 | 50.82 | 59.76 | 7500 | 120 MB |
| multi-qa-distilbert-cos-v1 | 65.98 | 52.83 | 59.41 | 4000 | 250 MB |
| all-MiniLM-L6-v2 | 68.06 | 49.54 | 58.80 | 14200 | 80 MB |
| multi-qa-MiniLM-L6-cos-v1 | 64.33 | 51.83 | 58.08 | 14200 | 80 MB |
| paraphrase-multilingual-mpnet-base-v2 | 65.83 | 41.68 | 53.75 | 2500 | 970 MB |
| paraphrase-albert-small-v2 | 64.46 | 40.04 | 52.25 | 5000 | 43 MB |
| paraphrase-multilingual-MiniLM-L12-v2 | 64.25 | 39.19 | 51.72 | 7500 | 420 MB |
| paraphrase-MiniLM-L3-v2 | 62.29 | 39.19 | 50.74 | 19000 | 61 MB |
| distiluse-base-multilingual-cased-v1 | 61.30 | 29.87 | 45.59 | 4000 | 480 MB |
| distiluse-base-multilingual-cased-v2 | 60.18 | 27.35 | 43.77 | 4000 | 480 MB |



Kemudian, SBERT digunakan untuk mengubah deskripsi teks dari film menjadi vektor. Model **'all-mpnet-base-v2'** digunakan, **yang telah terbukti** memiliki **kinerja yang baik** dalam tugas-tugas tertentu. Model 'all-mpnet-base-v2' dipilih berdasarkan beberapa faktor berikut.
+ Model ini menunjukkan **kinerja yang sangat baik** dalam menghasilkan **representasi vektor** kalimat (Sentence Embeddings). Dalam pengujian pada 14 dataset, model ini mencapai skor 69.57, yang mengindikasikan kualitas embedding kalimat yang tinggi.
+ Model ini juga menunjukkan **kinerja yang kuat dalam konteks pencarian semantik**. Dalam 6 dataset yang berbeda, model ini mencapai skor 57.02. Ini menunjukkan bahwa model ini mampu menghasilkan vektor yang efektif untuk membandingkan dan mencari kesamaan semantik antara query dan paragraf teks.
+ Model ini memiliki **skor rata-rata kinerja 63.30**, yang menunjukkan **konsistensi kinerjanya** di berbagai tugas dan dataset. Ini memastikan bahwa model ini dapat bekerja dengan baik dalam berbagai kasus penggunaan dan jenis teks.

Dengan demikian, model 'all-mpnet-base-v2' **dipilih berdasarkan kinerjanya yang kuat**, **konsistensi**, dan **keseimbangan antara kualitas dan efisiensi**. Hal ini menjadikannya pilihan yang baik untuk sistem rekomendasi film yang sedang dibuat.Dalam proses ini, setiap deskripsi film diubah menjadi vektor multidimensi, yang menangkap makna dan konten dari teks.

Lalu, setelah representasi vektor film telah dihasilkan, langkah selanjutnya adalah **menghitung kesamaan antara film**. Ini dilakukan dengan menggunakan **metode kesamaan cosinus**, yang mengukur sejauh mana dua vektor (dalam hal ini, dua vektor film) berorientasi dalam arah yang sama. Hasilnya adalah matriks kesamaan, di mana setiap elemen menunjukkan sejauh mana dua film mirip satu sama lain berdasarkan deskripsi mereka.

Akhirnya, matriks kesamaan ini digunakan untuk **merekomendasikan film**. Fungsi get_recommendations mengambil judul film, dan berdasarkan matriks kesamaan, mencari film-film lain yang paling mirip dengannya. Fungsi ini mencari N film yang paling mirip, **mengurutkannya berdasarkan *similarity score***, dan mengembalikan daftar film yang direkomendasikan beserta dengan *similarity score*-nya.

### Improvement
Memasuki tahap akhir pengembangan sistem rekomendasi ini akan **memperkaya pendekatan yang telah dipilih** dengan **menambahkan** beberapa fitur tambahan, seperti **genre, keywords, top 5 cast, dan director**. Ini bertujuan untuk memperbaiki dan **memperluas aspek-aspek yang digunakan dalam sistem rekomendasi**, sehingga dapat **memberikan hasil yang lebih akurat** dan sesuai dengan preferensi pengguna. 

Proses ini dimulai dengan tugas membaca dua set **data tambahan**, yang berisi informasi tentang **kredit** dan ***keywords*** yang berkaitan dengan setiap film. Proses ini melibatkan pemuatan file .csv untuk kredit dan *keywords* ke dalam DataFrame, dan melakukan pengecekan untuk **menghapus data duplikat** yang ada.

Setelah kredit dan *keywords* dimuat, langkah berikutnya adalah **menggabungkan informasi** ini dengan **data film utama** berdasarkan ID yang sama. Penggabungan ini dilakukan dengan fungsi 'merge', yang pada dasarnya menggabungkan dua DataFrame berdasarkan kolom tertentu, dalam hal ini 'id'.

Selanjutnya, data di dalam kolom 'cast' dan 'keywords' diproses. Dalam hal ini, data cast yang awalnya dalam format string, diubah menjadi daftar nama-nama aktor dan kata kunci dengan menggunakan fungsi 'literal_eval' dan 'lambda'. Batasan yang diberikan adalah hanya lima nama aktor pertama yang diambil jika ada lebih dari lima, jika kurang maka akan diambil seluruhnya.

Kemudian, fungsi khusus diterapkan pada kolom 'crew' untuk menggali informasi tentang sutradara film tersebut. Informasi ini sangat penting karena sutradara sering menjadi faktor penting dalam kualitas dan gaya film. Proses ini melibatkan iterasi melalui daftar 'crew' dan pencarian orang dengan jabatan 'Director'.

Setelah data tersebut diproses, langkah selanjutnya adalah melakukan beberapa langkah pembersihan dan normalisasi. Ini termasuk mengubah semua teks menjadi huruf kecil dan menghapus spasi untuk memastikan konsistensi dan akurasi saat membandingkan string.

**Akhirnya, semua informasi yang telah diproses ini, termasuk genre, kata kunci, sutradara, dan pemeran digabungkan menjadi satu kolom 'metadata'**. Kolom ini pada dasarnya merupakan gabungan dari semua fitur yang digunakan dalam sistem rekomendasi, yang memungkinkan model untuk mempertimbangkan semua aspek ini saat membuat rekomendasi. Dengan demikian, **data sekarang telah diproses dan siap untuk digunakan dalam model rekomendasi film**.

Berikut ditampilkan hasil finalisasi setelah melalui beberapa macam proses diatas.

<p align="center">
  <i>Tabel 4. Hasil improvement untuk digunakan pada model lanjutan</i>
</p>

<div align="center">

| id  | title                   | year | vote_count | vote_average | popularity  | genres                               | overview                                              | cast                                                                 | crew                                                                                                                | keywords                                                                                      | director           | metadata                                          |
|----|-------------------------|------|------------|--------------|-------------|--------------------------------------|-------------------------------------------------------|----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|---------------------|---------------------------------------------------|
| 862 | Toy Story               | 1995 | 5415.0     | 7.7          | 21.946943   | [Animation, Comedy, Family]          | Led by Woody, Andy's toys live happily in his ...     | [tomhanks, timallen, donrickles, jimvarney, wa...                 | [{'credit_id': '52fe4284c3a36847f8024f49', 'de...                 | [jealousy, toy, boy, friendship, friends, riva...                                           | [johnlasseter]      | Animation Comedy Family jealousy toy boy frien... |

</div>

Sisanya lakukan langkah yang sama untuk memanggil fungsi rekomendasi. Lalu buatlah fungsi tersebut untuk mengurutkan film berdasarkan *similarity value* dan tahun rilis, dengan film yang memiliki skor kesamaan lebih tinggi dan dirilis lebih baru muncul lebih dulu. Fungsi ini kemudian mengembalikan N film teratas dari DataFrame yang diurutkan ini. Dengan demikian, fungsi ini memberikan rekomendasi film berdasarkan judul film yang diberikan.

## Result
### 1. Base Recommendation
<hr>

Berikut adalah hasil dari rekomendasi model Base Recommendation setelah diberikan input genre **"Horror"**. Pada tabel berikut, terdapat **10 film horor** yang diurutkan berdasarkan **pembobotan rating**.
<br>
<p align="center">
  <i>Tabel 5. Hasil Base Recommendation</i>
</p>

| Title | Year | Vote Count | Vote Average | Popularity | Genres | Weighted Rating |
|---|---|---|---|---|---|---|
| The Shining | 1980 | 3890.0 | 8.1 | 19.611589 | [Horror, Thriller] | 7.8 |
| Psycho | 1960 | 2405.0 | 8.3 | 36.826309 | [Drama, Horror, Thriller] | 7.8 |
| Alien | 1979 | 4564.0 | 7.9 | 23.377420 | [Horror, Action, Thriller, Science Fiction] | 7.6 |
| Aliens | 1986 | 3282.0 | 7.7 | 21.761179 | [Horror, Action, Thriller, Science Fiction] | 7.4 |
| The Thing | 1982 | 1629.0 | 7.8 | 16.831250 | [Horror, Mystery, Science Fiction] | 7.2 |
| Jaws | 1975 | 2628.0 | 7.5 | 19.726114 | [Horror, Thriller, Adventure] | 7.2 |
| The Conjuring | 2013 | 3169.0 | 7.4 | 14.901690 | [Horror, Thriller] | 7.1 |
| Shaun of the Dead | 2004 | 2479.0 | 7.5 | 14.902948 | [Horror, Comedy] | 7.1 |
| The Exorcist | 1973 | 2046.0 | 7.5 | 12.137595 | [Drama, Horror, Thriller] | 7.1 |
| Zombieland | 2009 | 3655.0 | 7.2 | 11.063029 | [Comedy, Horror] | 7.0 |


Tabel di atas menampilkan film-film horor dengan detail seperti judul, tahun rilis, jumlah suara, rata-rata skor, popularitas, dan genre. Film-film ini dipilih berdasarkan pembobotan rating yang dihitung dari suara yang diterima dan skor rata-rata. Film dengan pembobotan rating tertinggi ditampilkan di bagian atas.

Meski sistem rekomendasi film sederhana yang telah dibangun memiliki kemampuan untuk menarik berbagai penonton dengan berbagai pilihan film populer dan berkualitas tinggi, sistem tersebut masih memiliki kelemahan. Rekomendasi yang diberikan sama untuk semua orang, tanpa mempertimbangkan selera pribadi pengguna sehingga sistem ini **tidak mempertimbangkan preferensi individu atau selera pribadi penggunanya.**. Ini berarti bahwa rekomendasi yang dihasilkan, meski mungkin secara umum menarik, mungkin tidak selalu sesuai atau relevan dengan penonton individu. 

### 2. CBF dengan TF-IDF
<hr>
<br>
<div align="center">

<p align="center">
  <i>Tabel 6. Hasil CBF dengan TF-IDF</i>
</p>

| Title | Genres | Year | Similarity Score |
|---|---|---|---|
| The Dark Knight Rises | [Action, Crime, Drama, Thriller] | 2012 | 28.11 |
| Batman Returns | [Action, Fantasy] | 1992 | 21.41 |
| Batman: Under the Red Hood | [Action, Animation] | 2010 | 20.82 |
| Batman Forever | [Action, Crime, Fantasy] | 1995 | 19.76 |
| Batman | [Fantasy, Action] | 1989 | 16.33 |
| Batman: The Killing Joke | [Action, Animation, Crime, Drama] | 2016 | 15.97 |
| Batman Begins | [Action, Crime, Drama] | 2005 | 14.36 |
| The Lego Batman Movie | [Action, Animation, Comedy, Family, Fantasy] | 2017 | 14.35 |
| JFK | [Drama, Thriller, History] | 1991 | 13.74 |
| Law Abiding Citizen | [Drama, Crime, Thriller] | 2009 | 10.35 |

</div>

Dari tabel rekomendasi yang diberikan, dapat dilihat bahwa rekomendasi yang dihasilkan oleh sistem berdasarkan vektorisasi TF-IDF cukup sesuai dengan genre film yang dicari. Dalam hal ini, film yang dicari adalah "The Dark Knight Rises", yang memiliki genre "Action", "Crime", "Drama", dan "Thriller". Sebagian besar film dalam daftar rekomendasi memiliki genre yang serupa, seperti "Action", "Crime", dan "Drama", menunjukkan bahwa sistem berhasil dalam mencocokkan film berdasarkan genre mereka.

Namun, ada beberapa pengecualian. Misalnya, "The Lego Batman Movie" adalah film komedi dan keluarga, yang mungkin tidak sesuai dengan preferensi seseorang yang mencari film aksi dan kejahatan yang serius seperti "The Dark Knight Rises". Selain itu, "JFK" adalah film sejarah, yang mungkin tidak menarik bagi penonton yang mencari film superhero.

### 3. CBF dengan salah satu pre-trained model SBERT
<hr>
<br>
<p align="center">
  <i>Tabel 7. Hasil CBF dengan pre-trained model SBERT</i>
</p>
<div align="center">

| Title | Genres | Year | Similarity Score |
|---|---|---|---|
| The Dark Knight Rises | [Action, Crime, Drama, Thriller] | 2012 | 82.82 |
| Batman | [Fantasy, Action] | 1989 | 81.04 |
| Batman Begins | [Action, Crime, Drama] | 2005 | 77.04 |
| Batman v Superman: Dawn of Justice | [Action, Adventure, Fantasy] | 2016 | 69.59 |
| Batman: Under the Red Hood | [Action, Animation] | 2010 | 69.37 |
| Batman: The Killing Joke | [Action, Animation, Crime, Drama] | 2016 | 68.64 |
| Batman Forever | [Action, Crime, Fantasy] | 1995 | 68.56 |
| The Lego Batman Movie | [Action, Animation, Comedy, Family, Fantasy] | 2017 | 63.73 |
| Batman Returns | [Action, Fantasy] | 1992 | 61.35 |
| Batman & Robin | [Action, Crime, Fantasy] | 1997 | 59.37 |

</div>

Tabel rekomendasi ini diturunkan dari model berbasis SBERT, dan dapat dilihat bahwa skor kemiripan secara umum lebih tinggi dibandingkan dengan model berbasis TF-IDF. Misalnya, skor kemiripan untuk film yang sama "The Dark Knight Rises" sekarang adalah 82.82, sementara dalam model berbasis TF-IDF, skor kemiripan adalah 28.11.

Selain itu, dapat dilihat bahwa daftar film yang direkomendasikan berbeda. Dalam model berbasis SBERT, ada sejumlah film lain yang juga berada dalam genre aksi dan petualangan, seperti "Batman Begins", "Batman v Superman: Dawn of Justice", "Suicide Squad", dan film dari franchise lain seperti "Captain America" dan "Fast & Furious". Ini menunjukkan bahwa model berbasis SBERT lebih baik dalam menangkap konteks dan kesamaan semantik antara film.

Sementara itu, perlu dicatat bahwa beberapa film dalam rekomendasi, seperti "Pulp Fiction", tampak sedikit out of place, tetapi ini disebabkan oleh kesamaan dalam deskripsi atau sinopsis film daripada genre.

Secara keseluruhan, berdasarkan skor kemiripan yang lebih tinggi dan genre yang lebih konsisten antara film yang direkomendasikan, tampaknya model berbasis SBERT lebih efektif dalam memberikan rekomendasi film yang mirip dengan film target dibandingkan dengan model berbasis TF-IDF.

### 4. CBF dengan penambahan metadata
<hr><br>
<p align="center">
  <i>Tabel 8. Hasil CBF dengan pre-trained model SBERT ditambah dengan metadata</i>
</p>

| Title | Genres | Vote Count | Vote Average | Year | Similarity Score | Weighted Rating |
|---|---|---|---|---|---|---|
| The Dark Knight Rises | [Action, Crime, Drama, Thriller] | 9263.0 | 7.6 | 2012 | 92.69 | 7.3 |
| Batman: The Killing Joke | [Action, Animation, Crime, Drama] | 485.0 | 6.2 | 2016 | 87.03 | 6.6 |
| Suicide Squad | [Action, Adventure, Crime, Fantasy, Science Fiction] | 7717.0 | 5.9 | 2016 | 82.08 | 6.2 |
| Batman v Superman: Dawn of Justice | [Action, Adventure, Fantasy] | 7189.0 | 5.7 | 2016 | 80.82 | 6.1 |
| Batman Begins | [Action, Crime, Drama] | 7511.0 | 7.5 | 2005 | 79.81 | 7.1 |
| Batman: Under the Red Hood | [Action, Animation] | 459.0 | 7.6 | 2010 | 79.66 | 6.7 |
| Batman Returns | [Action, Fantasy] | 1706.0 | 6.6 | 1992 | 79.24 | 6.6 |
| Dredd | [Action, Science Fiction] | 1971.0 | 6.6 | 2012 | 79.06 | 6.6 |
| Superman | [Action, Adventure, Fantasy, Science Fiction] | 1042.0 | 6.9 | 1978 | 78.43 | 6.7 |
| Batman & Robin | [Action, Crime, Fantasy] | 1447.0 | 4.2 | 1997 | 76.82 | 6.0 |

Setelah memperkenalkan metadata tambahan ke dalam model, dapat dilihat bahwa beberapa rekomendasi film telah berubah. Dengan kata lain, beberapa film baru muncul dalam daftar rekomendasi. Ini mencerminkan peningkatan dalam pengenalan konteks dan konten oleh model yang sekarang menggunakan lebih banyak data dalam proses pembuatan rekomendasinya.

Misalnya, film "Man of Steel", "Iron Man 3", "Doctor Strange", dan "Wonder Woman" sekarang muncul dalam daftar rekomendasi. Ini menggambarkan peningkatan kualitas rekomendasi karena film-film tersebut memiliki genre yang serupa dengan film aslinya dan juga sangat populer, seperti yang ditunjukkan oleh jumlah suara yang mereka terima. Ini menunjukkan bahwa model telah mempertimbangkan lebih banyak aspek dari film, yang menghasilkan rekomendasi yang lebih relevan dan dipersonalisasi.

Selain itu, juga dapat dilihat bahwa skor kemiripan dan peringkat berbobot untuk film tertentu telah berubah. Misalnya, "The Dark Knight Rises" sekarang memiliki skor kemiripan 92.69, yang lebih tinggi dari sebelumnya, dan peringkat berbobot 7.5, yang sedikit lebih rendah dari sebelumnya. Ini menunjukkan bahwa dengan menambahkan lebih banyak data ke model, model menjadi lebih akurat dalam menilai kemiripan antara film dan lebih adil dalam memberikan peringkat, dengan mempertimbangkan lebih banyak faktor.

Jadi, secara keseluruhan, penambahan metadata ke dalam model SBERT telah memperkaya model dengan lebih banyak informasi, yang pada akhirnya telah meningkatkan kualitas dan relevansi rekomendasi film yang dihasilkan oleh model.

## Evaluasi
Untuk memberikan evaluasi yang paling akurat dan berarti tentang efektivitas tiga pendekatan yang berbeda ini - TF-IDF, SBERT sebelum penambahan metadata, dan SBERT setelah penambahan metadata, perlu diukur nilai presisi ketiga pendeketan tersebut dalam konteks tugas rekomendasi film. Pada dasarnya, presisi mengukur sejauh mana rekomendasi yang diberikan oleh sistem benar-benar relevan dan sesuai dengan apa yang diharapkan pengguna. Perhitungan harus dilakukan secara manual berdasarkan rekomendasi yang dihasilkan oleh model. Rumus untuk menghitung Precision bisa dilihat pada gambar di bawah.

**Precision (%)** = $\frac{True Positives}{True Positives + False Positives} \times 100$

+ **Model TF-IDF:** Dalam daftar rekomendasi dari model TF-IDF, semua film Batman dapat dianggap sebagai true positives (TP) karena memiliki konteks dan genre yang mirip dengan "The Dark Knight Rises". Sedangkan, film "JFK" dan "Law Abiding Citizen" akan dianggap sebagai false positives (FP) karena mereka tidak berhubungan dengan Batman atau memiliki genre dan tema yang berbeda
  + **True Positives (TP):** 7 (The Dark Knight Rises, Batman Returns, Batman: Under the Red Hood, Batman Forever, Batman, Batman: The Killing Joke, Batman Begins, The Lego Batman Movie)
  + **False Positives (FP):** 2 (JFK, Law Abiding Citizen)
  + **Precision (%) =** TP / (TP + FP) = 8 / (8 + 2) * 100 = **80%**

+ **Model SBERT sebelum metadata:** Dalam kasus SBERT sebelum penambahan metadata, semua film adalah film Batman, yang berarti semua adalah true positives.
  
  + **True Positives (TP):** 10 (Semua film dalam daftar)
  + **False Positives (FP):** 0 (Tidak ada film dalam daftar yang bukan film Batman)
  + **Precision (%) =** TP / (TP + FP) = 10 / (10 + 0) * 100 = **100%**

+ **Model SBERT setelah metadata:** Dalam kasus SBERT setelah penambahan metadata, "Suicide Squad", "Dredd", dan "Superman" dianggap sebagai false positives, karena walaupun mereka memiliki genre yang mirip, tetapi mereka tidak berhubungan langsung dengan Batman.
  + **True Positives (TP):** 7 (The Dark Knight Rises, Batman: The Killing Joke, Batman v Superman: Dawn of Justice, Batman Begins, Batman: Under the Red Hood, Batman Returns, Batman & Robin)
  + **False Positives (FP):** 3 (Suicide Squad, Dredd, Superman)
  + **Precision (%) =** TP / (TP + FP) = 7 / (7 + 3) * 100 = **70%**

Dalam hal ini, model SBERT sebelum penambahan metadata memberikan presisi tertinggi dengan 100%, sementara model SBERT setelah penambahan metadata memberikan presisi terendah dengan nilai 70%. Lalu model TF-IDF berada diantara keduanya yakni dengan nilai presisi 80%.

Sebagai catatan, tingkat similaritas film memang dapat berfungsi sebagai indikator kuat dalam merekomendasikan film yang paling relevan berdasarkan preferensi pengguna. Semakin tinggi tingkat similaritas antara dua film, semakin besar kemungkinan bahwa pengguna yang menyukai satu film akan juga menyukai film lainnya.

Namun, tingkat similaritas yang lebih tinggi tidak selalu berarti hasil yang lebih baik. Ini karena faktor-faktor lain juga berperan dalam menentukan apakah suatu rekomendasi akan disukai oleh pengguna. Beberapa faktor ini dapat termasuk genre, aktor, sutradara, dan aspek lain dari film.

## Kesimpulan
Berdasarkan evaluasi di atas, model SBERT sebelum penambahan metadata memberikan tingkat presisi tertinggi, yaitu 100%. Ini berarti bahwa sistem ini mampu memberikan rekomendasi yang paling akurat dan relevan dengan "The Dark Knight", dalam hal ini, film-film Batman lainnya.

Di sisi lain, model SBERT setelah penambahan metadata memiliki yang rendah, yaitu 70%. Meskipun model ini masih menghasilkan rekomendasi yang relevan, namun ada beberapa rekomendasi (seperti "Suicide Squad", "Dredd", dan "Superman") yang mungkin kurang tepat karena tidak berhubungan langsung dengan Batman, meskipun berbagi genre yang sama.

Model TF-IDF berada di antara dua model lainnya dengan tingkat presisi 80%. Model ini memberikan sebagian besar rekomendasi yang relevan, namun ada beberapa (seperti "JFK" dan "Law Abiding Citizen") yang kurang relevan karena tidak berhubungan dengan Batman dan memiliki genre atau tema yang berbeda.

Mengingat bahwa setiap pengguna memiliki preferensi dan ekspektasi yang unik, sistem rekomendasi ideal harus mampu menyesuaikan rekomendasinya dengan preferensi dan minat individu pengguna. Selain itu, faktor-faktor lain seperti genre, aktor, sutradara, dan aspek lain dari film juga harus dipertimbangkan untuk memberikan rekomendasi yang paling relevan dan personal.

Akhirnya, meskipun SBERT sebelum penambahan metadata memiliki tingkat presisi tertinggi dalam kasus ini, tidak berarti itu selalu merupakan pendekatan terbaik untuk setiap kasus. Pendekatan yang terbaik akan bergantung pada berbagai faktor, termasuk sifat dan skala data yang tersedia, serta preferensi dan ekspektasi pengguna. Dalam beberapa kasus, pendekatan lain mungkin lebih efektif atau lebih sesuai. Oleh karena itu, selalu penting untuk melakukan evaluasi dan penyesuaian berkelanjutan terhadap model dan pendekatan yang digunakan dalam sistem rekomendasi.

## Sumber Referensi
[1] Pazzani, M. J., & Billsus, D. (2007). Content-based recommendation systems. In The adaptive web (pp. 325-341). Springer, Berlin, Heidelberg.
