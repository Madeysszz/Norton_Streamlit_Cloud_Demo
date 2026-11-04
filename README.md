# Norton Streamlit Cloud Demo

Dashboard demonstrasi empat halaman untuk deployment Streamlit Community Cloud. Seluruh 128 log, 16 akun, 64 referensi, waktu, nilai, dan skor dibuat secara sintetis dalam memori. Label IF, ECOD, dan LOF merupakan simulasi tampilan; aplikasi tidak melatih model atau menggunakan data transaksi perusahaan.

## File utama

- `app.py`: entrypoint empat halaman dan seluruh filter/interaksi.
- `demo_data.py`: generator deterministik dengan seed 2026 dan tanggal Januari 2030.
- `requirements.txt`: dependensi runtime NumPy, pandas, Streamlit, dan Plotly.
- `requirements-dev.txt`: dependensi runtime beserta pytest.
- `.streamlit/config.toml`: konfigurasi Streamlit.
- `.python-version`: deklarasi target pengembangan Python 3.12.
- `tests/test_dashboard.py`: 16 pengujian generator, empat halaman, filter, threshold, konteks akun, kondisi kosong, dan isolasi data privat.
- `scripts/smoke_test.py`: pemeriksaan health dan halaman utama pada server sementara.
- `.github/workflows/deployment-ci.yml`: CI otomatis untuk push/pull request ke main.

## Menjalankan secara lokal

Gunakan environment Python 3.12, kemudian jalankan dari root repository:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Tidak diperlukan dataset, database, secret, atau environment dari proyek lain. Aplikasi mengabaikan `NORTON_DATA_ROOT`. Tekan Ctrl+C pada terminal server untuk menghentikannya.

## Pengujian dan CI

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q app.py demo_data.py tests scripts
python -m pytest tests -q --junitxml=test-results/pytest.xml
python scripts/smoke_test.py
```

CI menjalankan syntax, 16 kasus pytest/AppTest, satu smoke test HTTP, job summary, dan unggahan laporan. [Lihat status GitHub Actions](https://github.com/Madeysszz/Norton_Streamlit_Cloud_Demo/actions). Tidak ada pipeline Kecerdasan Bisnis, training model, atau data Norton yang dijalankan dalam CI.

## Deployment Streamlit Community Cloud

Pilih Create app pada Streamlit Community Cloud, lalu gunakan:

| Pengaturan | Nilai |
| --- | --- |
| Repository | `Madeysszz/Norton_Streamlit_Cloud_Demo` |
| Branch | `main` |
| Main file path | `app.py` |
| Python pada Advanced settings | `3.12` |
| Dependency file | `requirements.txt` di root repository |
| Secrets | Kosong |

`.python-version` tidak menggantikan pilihan Python pada Advanced settings. Repository tetap private. Jika aplikasi hendak diakses publik, pemilik akun perlu menyetujui pengaturan visibilitas aplikasi hosting secara terpisah.

Hosting belum diverifikasi dan URL publik belum tersedia. Setelah deploy selesai, periksa banner sintetis, keempat halaman, seluruh filter, dan akses URL sesuai visibilitas yang dipilih.

Kode aplikasi dan generator disalin dari versi cloud-demo proyek [Kecerdasan Bisnis dan Deployment Aplikasi](https://github.com/Madeysszz/Kecerdasan_Bisnis_dan_Deployment_App_MBKMKonversi). Repository ini berdiri sendiri dan tidak memerlukan file dari proyek asal.
