# Workflow-CI — IBM Telco Customer Churn

Repository terpisah untuk retraining model melalui MLflow Project, penyimpanan artefak permanen pada repository ini, dan build/push image menggunakan MLflow. Identitas siswa: **Muhammad-Febrilian-Kurnia-Putra**, username Dicoding **febril_putra**.

```text
main: data preprocessing + selected_params.json
  |
  v
GitHub Actions -> mlflow run MLProject -> winner.json (run ID tepat)
                                            |
                 +--------------------------+----------------------+
                 v                                                 v
model-artifacts/artifacts/<RUN_ID>/                   mlflow models build-docker
model + reports + provenance                                      |
                 ^                                     uji prediksi container
                 |                                                 |
                 +------- deployment.json <--------------- Docker Hub
```

| Lokasi | Isi |
| --- | --- |
| `.github/workflows/training.yml` | Trigger push perubahan training pada main atau manual; environment, retraining, persistensi, image, dan smoke test |
| `MLProject/MLproject` | Konfigurasi MLflow; nama berkas harus menggunakan huruf `p` kecil |
| `MLProject/conda.yaml` | Spesifikasi environment Python 3.12.10 dan dependency tersemat |
| `MLProject/modelling.py` | Manual logging untuk retraining parameter pilihan; berasal dari script tuning proyek lokal |
| `MLProject/training_common.py` | Validasi data, metrik, artefak, dan signature model |
| `MLProject/selected_params.json` | Parameter pemenang yang diekspor tuning lokal; bukan parameter yang dipilih dari skor test |
| `MLProject/telco_churn_preprocessing/` | Snapshot lengkap hasil preprocessing dengan manifest checksum |
| Branch `model-artifacts`, `artifacts/<RUN_ID>/` | Model MLflow lengkap, laporan, transformer, provenance, serta bukti deployment setelah push berhasil |

Data berasal dari [repository eksperimen](https://github.com/febrilputra/Eksperimen_SML_Muhammad-Febrilian-Kurnia-Putra). Script CI menggunakan snapshot data yang dicommit pada repository ini sehingga checkout dapat berjalan mandiri. Saat memperbarui data, salin seluruh folder preprocessing beserta manifest; jangan mencampur versi file. `training_provenance.json` mencatat checksum input dan manifest sumber.

Siapkan repository publik dan izinkan workflow menulis contents. Masukkan dua GitHub Actions secrets berikut melalui pengaturan repository:

| Secret | Isi |
| --- | --- |
| `DOCKERHUB_USERNAME` | `febrilputra17` |
| `DOCKERHUB_TOKEN` | Access token dengan hak push ke repository `telco-churn` milik akun tersebut |

Tidak ada token di kode. Workflow tetap menyimpan model pada branch `model-artifacts` sebelum memeriksa secrets Docker Hub. Jika secret belum tersedia, workflow berhenti dengan pesan yang jelas dan status gagal; kondisi tersebut belum memenuhi Advanced CI. Pastikan repository image di Docker Hub publik agar reviewer dapat mengaksesnya.

Jalankan lokal dari root repository, dengan environment yang telah menginstal `MLProject/requirements.txt`:

```powershell
New-Item -ItemType Directory -Force runtime | Out-Null
$env:MLFLOW_TRACKING_URI = ([System.Uri](Join-Path (Resolve-Path runtime).Path 'mlruns')).AbsoluteUri
$env:MLFLOW_EXPERIMENT_NAME = 'telco-churn-ci-local-check'
mlflow run ./MLProject --env-manager=local -P output_dir=../runtime/current
```

Pada struktur kerja proyek ini, argumen output dan tracking diarahkan ke `submission/runtime` agar log terpusat; gunakan `output_dir=../../../runtime/ci-local` karena proses training berjalan dari `MLProject`. Perintah di atas juga dapat digunakan ketika repository di-clone mandiri. Path relatif menghindari masalah escaping path Windows yang mengandung tanda seru pada MLflow 2.19. `--env-manager=local` sengaja digunakan setelah dependency tersemat diinstal; konfigurasi Conda tetap tersedia jika ingin menjalankan default environment MLflow.

Workflow mengambil run ID langsung dari `runtime/current/winner.json`, kemudian model yang persis sama diunduh melalui API artefak. Tidak ada pencarian run terbaru yang dapat tertukar dengan proses lain. CI hanya melatih parameter yang sebelumnya dipilih; evaluasi test adalah pemeriksaan final model hasil retraining, bukan dasar perubahan parameter otomatis.

Artefak disimpan pada branch `model-artifacts` di repository yang sama. Branch itu tidak termasuk trigger push sehingga commit output tidak membentuk perulangan. Workflow memeriksa ukuran per berkas di bawah 95 MiB sebelum commit biasa; jika model membesar, konfigurasi LFS harus dilakukan lebih dahulu. Actions artifact dengan masa simpan 30 hari merupakan salinan tambahan, bukan penyimpanan utama.

Image memakai tag unik `<COMMIT_SHA>-<GITHUB_RUN_NUMBER>-<GITHUB_RUN_ATTEMPT>` agar percobaan ulang tidak menimpa tag percobaan sebelumnya. Sebelum push, smoke test menunggu layanan sehat hingga 180 detik, termasuk bila koneksi sementara di-reset saat container baru mulai berjalan. Setelah sehat, smoke test mengirim tiga baris validation ke `/invocations` dan membandingkan hasil dengan prediksi model yang tersimpan. Setelah berhasil, `deployment.json` mencatat nama/tag image, URL Docker Hub, commit, run ID, dan checksum model. `smoke_test.json` merekam request dan response aktual. `MLProject/DockerHub.txt` harus diperbarui dengan tautan/tag aktual yang dipakai untuk submission.

Build menggunakan `mlflow models build-docker --env-manager local`. MLflow membuat image `python:3.12.10-slim` dan memasang dependency dari model ke dalam image; opsi ini tidak bergantung pada environment Python host saat container berjalan. Atribut Git mempertahankan byte dataset dan artefak biner agar checksum tidak berubah akibat konversi CRLF/LF. Provenance juga menyimpan SHA256 source code aktual; tag commit repository saja belum mewakili source yang belum dicommit saat eksperimen lokal.

Unduh direktori artefak berdasarkan run yang dipilih dari branch `model-artifacts`, atau pull nama image persis dalam `deployment.json`. Contoh serving setelah mengganti nilai `IMAGE` dengan image aktual:

```powershell
docker pull IMAGE
docker run --rm -p 5001:8080 IMAGE
```

Port model MLflow dalam container adalah 8080. Endpoint prediksi menerima fitur numerik sesuai signature melalui `dataframe_split`; sistem monitoring melakukan preprocessing raw menggunakan transformer dan skema dari run yang sama.

Kode dan konfigurasi belum membuktikan pekerjaan online selesai. Bukti yang diperlukan: setidaknya satu Actions run berhasil, artefak permanen dapat diunduh, image dapat di-pull, dan serving/monitoring lokal menggunakan image tersebut.

Referensi: [MLflow Projects 2.19](https://mlflow.org/docs/2.19.0/projects.html), [MLflow build-docker 2.19](https://mlflow.org/docs/2.19.0/cli.html#build-docker), [penyimpanan Actions artifacts](https://docs.github.com/en/actions/tutorials/store-and-share-data).
