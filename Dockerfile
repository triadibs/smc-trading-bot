# Dockerfile untuk Python Trading Bot

# Tahap 1: Tentukan Base Image
# Gunakan base image Python yang ringan dan efisien
FROM python:3.11-slim

# Tahap 2: Tetapkan Direktori Kerja
# Tetapkan direktori kerja di dalam container agar rapi
WORKDIR /app

# Tahap 3: Instal Dependensi
# Salin file requirements terlebih dahulu untuk memanfaatkan Docker layer caching
COPY requirements.txt .
# Instal semua library yang dibutuhkan dan jangan simpan cache untuk menjaga ukuran image
RUN pip install --no-cache-dir -r requirements.txt

# Tahap 4: Salin Kode Aplikasi
# Sekarang, salin sisa kode proyek Anda (termasuk smc_alerter.py) ke dalam container
COPY . .

# Tahap 5: Tentukan Perintah Start
# Perintah yang akan dijalankan secara otomatis saat container dimulai
CMD ["python", "smc_trading_bot.py"]