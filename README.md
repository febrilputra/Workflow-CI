# Workflow-CI — IBM Telco Customer Churn

Repository terpisah untuk retraining model melalui MLflow Project, penyimpanan artefak permanen pada repository ini, dan build/push image menggunakan MLflow. Identitas siswa: **Muhammad-Febrilian-Kurnia-Putra**, username Dicoding **febril_putra**.

[GitHub Actions run 34599549644](https://github.com/febrilputra/Workflow-CI/actions/runs/34599549644) berhasil pada 11 September 2026. Run tersebut melatih model, menyimpan artefak pada repository ini, membangun image dengan MLflow, menguji prediksi container, dan melakukan push ke [Docker Hub](https://hub.docker.com/r/febrilputra17/telco-churn).

| Bukti pipeline terpilih | Identitas |
| --- | --- |
| Commit source training | `31d149970d6295e53b019140fe9f42a062a208eb` |
| Run MLflow | `2468308e231f4eca8152e387280e10a8` |
| Image hasil CI | `febrilputra17/telco-churn:31d149970d6295e53b019140fe9f42a062a208eb-2-1` |
| Artefak permanen | [Snapshot pada commit artefak 6cc545e](https://github.com/febrilputra/Workflow-CI/tree/6cc545e1b041f4cd1482a6378ed4adaf810550e4/artifacts/2468308e231f4eca8152e387280e10a8) |
| Integritas unduhan | 25 checksum berkas pada `artifact_checksums.json` cocok dengan unduhan lokal |
| Smoke test CI | Prediksi `[0, 1, 0]` sama dengan model tersimpan; receipt pada `smoke_test.json` |

Tautan artefak menunjuk commit tertentu agar tetap merujuk hasil run yang sama setelah branch `model-artifacts` diperbarui oleh training berikutnya. `deployment.json` merekam image, commit source, run ID, dan SHA256 model.

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

Repository GitHub dan Docker Hub yang digunakan berstatus publik. Untuk menjalankan workflow pada repository lain, izinkan workflow menulis contents dan masukkan dua GitHub Actions secrets berikut melalui pengaturan repository:

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

Unduh direktori artefak berdasarkan snapshot commit di atas, atau pull image terpilih berikut:

```powershell
docker pull febrilputra17/telco-churn:31d149970d6295e53b019140fe9f42a062a208eb-2-1
docker run --rm -p 5001:8080 febrilputra17/telco-churn:31d149970d6295e53b019140fe9f42a062a208eb-2-1
```

Port model MLflow dalam container adalah 8080. Endpoint prediksi menerima fitur numerik sesuai signature melalui `dataframe_split`; sistem monitoring melakukan preprocessing raw menggunakan transformer dan skema dari run yang sama.

Keberhasilan pipeline online ditunjukkan oleh run, artefak permanen, dan image di atas. Status serving serta monitoring lokal atas image tersebut dicatat dalam folder `Monitoring dan Logging` pada paket submission.

Referensi: [MLflow Projects 2.19](https://mlflow.org/docs/2.19.0/projects.html), [MLflow build-docker 2.19](https://mlflow.org/docs/2.19.0/cli.html#build-docker), [penyimpanan Actions artifacts](https://docs.github.com/en/actions/tutorials/store-and-share-data).
