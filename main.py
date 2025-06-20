import os
import asyncio
import ccxt.async_support as ccxt
import nest_asyncio
import json
import pandas as pd
from telegram import Bot
import google.generativeai as genai

# Menerapkan nest_asyncio agar asyncio bisa berjalan di lingkungan seperti notebook
nest_asyncio.apply()

# --- 1. Konfigurasi & Inisialisasi ---

# Memuat variabel dari environment (pastikan Anda sudah mengaturnya)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')

# Memastikan semua variabel environment telah diisi
assert all([GEMINI_API_KEY, TELEGRAM_TOKEN, CHAT_ID]), "Variabel ENV (GEMINI_API_KEY, TELEGRAM_TOKEN, CHAT_ID) belum lengkap!"

# Inisialisasi layanan
genai.configure(api_key=GEMINI_API_KEY)
bot = Bot(token=TELEGRAM_TOKEN)
exchange = None # Akan diinisialisasi di dalam main()

# --- 2. Pengaturan Strategi ---

SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'DOGE/USDT', 'ADA/USDT']  # Simbol yang akan dianalisis
TIMEFRAME = '15m'                     # Timeframe yang digunakan
EXCHANGE_NAME = 'kraken'             # Ganti dengan exchange pilihan Anda (kraken, bybit, dll)
CANDLES = 200                         # Jumlah candle yang diambil untuk analisis
INTERVAL = 300                        # Jeda waktu antar siklus analisis dalam detik (300 detik = 5 menit)

# Dictionary untuk melacak notifikasi yang sudah terkirim
# Akan menyimpan ID unik seperti: 'BTC/USDT_OB_1672531200000'
alerted = {}

# --- 3. Fungsi Deteksi ---

def find_fvg(ohlcv: list):
    """Mendeteksi Fair Value Gap (FVG) terbaru dari data OHLCV."""
    for i in range(len(ohlcv) - 3, 0, -1):
        c1_high = ohlcv[i][2]
        c3_low = ohlcv[i + 2][3]
        
        # Bullish FVG (Gap antara high candle pertama dan low candle ketiga)
        if c3_low > c1_high:
            return {'type': 'Bullish', 'min': c1_high, 'max': c3_low, 't': ohlcv[i + 1][0]}
            
        c1_low = ohlcv[i][3]
        c3_high = ohlcv[i + 2][2]

        # Bearish FVG (Gap antara low candle pertama dan high candle ketiga)
        if c3_high < c1_low:
            return {'type': 'Bearish', 'min': c3_high, 'max': c1_low, 't': ohlcv[i + 1][0]}
    return None

def find_order_block(df: pd.DataFrame):
    """Mendeteksi Order Block sederhana berdasarkan candle dengan body besar."""
    body_threshold = 0.7  # Badan candle harus > 70% dari total range-nya
    for i in range(len(df) - 3, 0, -1):
        o, h, l, c = df.loc[i, ['o', 'h', 'l', 'c']]
        body = abs(c - o)
        price_range = h - l
        
        if price_range == 0: continue # Hindari pembagian dengan nol jika candle doji
        
        if (body / price_range) > body_threshold:
            # Menggunakan definisi sederhana: candle bullish besar = OB bullish
            if c > o:
                return {'type': 'Bullish', 'min': o, 'max': c, 't': df.loc[i, 't']}
            else:
                return {'type': 'Bearish', 'min': c, 'max': o, 't': df.loc[i, 't']}
    return None

# --- 4. Fungsi Analisis Inti (Telah Direvisi Total) ---

async def analyze(symbol: str):
    """
    Fungsi inti yang menjalankan seluruh alur analisis untuk satu simbol.
    Mendeteksi OB dan FVG secara terpisah dan mengirim notifikasi jika harga masuk ke salah satunya.
    """
    try:
        print(f"🔍 Menganalisis {symbol} pada timeframe {TIMEFRAME}...")

        # Ambil data harga terbaru
        ohlcv = await exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=CANDLES)
        df = pd.DataFrame(ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])
        current_price = df['c'].iloc[-1]

        # --- Bagian 1: Pengecekan Order Block (OB) ---
        ob = find_order_block(df)
        if ob:
            ob_alert_id = f"{symbol}_OB_{ob['t']}" # ID unik untuk alert ini
            
            # Cek jika harga masuk OB DAN alert untuk OB ini belum pernah dikirim
            if (ob['min'] <= current_price <= ob['max']) and not alerted.get(ob_alert_id):
                print(f"🔥 [MATCH OB] Harga {symbol} ({current_price}) masuk ke Order Block.")
                
                # Buat prompt yang spesifik untuk AI
                prompt = f"""SYSTEM: Anda adalah seorang analis Smart Money Concepts (SMC) profesional. Berikan analisis dan sinyal trading berdasarkan data berikut.
USER:
- Simbol: {symbol}
- Timeframe: {TIMEFRAME}
- Pemicu Sinyal: Harga telah memasuki zona Order Block.
- Tipe Order Block: {ob['type']}
- Zona Order Block: {ob['min']} - {ob['max']}
- Harga Saat Ini: {current_price}

Berikan output dalam format JSON dengan kunci "keputusan" (isinya "LONG", "SHORT", atau "WAIT") dan "alasan" (penjelasan singkat).
Contoh: {{"keputusan": "LONG", "alasan": "Harga bereaksi di OB bullish dan menunjukkan potensi rejection ke atas."}}"""

                # Panggil AI untuk analisis
                model = genai.GenerativeModel('gemini-1.5-flash-latest')
                response = await model.generate_content_async(prompt)
                
                # Kirim notifikasi ke Telegram
                await bot.send_message(CHAT_ID, f"🔥 **Sinyal Potensial (Order Block)** 🔥\n\n*Simbol:* `{symbol}`\n*Timeframe:* `{TIMEFRAME}`\n*Harga Masuk:* `{current_price}`\n*Zona OB ({ob['type']}):* `{ob['min']} - {ob['max']}`\n\n*Analisis AI:* \n```{response.text.strip()}```")
                
                alerted[ob_alert_id] = True # Tandai alert ini sudah terkirim

        # --- Bagian 2: Pengecekan Fair Value Gap (FVG) ---
        fvg = find_fvg(ohlcv)
        if fvg:
            fvg_alert_id = f"{symbol}_FVG_{fvg['t']}" # ID unik untuk alert ini
            
            # Cek jika harga masuk FVG DAN alert untuk FVG ini belum pernah dikirim
            if (fvg['min'] <= current_price <= fvg['max']) and not alerted.get(fvg_alert_id):
                print(f"⚡️ [MATCH FVG] Harga {symbol} ({current_price}) masuk ke Fair Value Gap.")
                
                prompt = f"""SYSTEM: Anda adalah seorang analis Smart Money Concepts (SMC) profesional. Berikan analisis dan sinyal trading berdasarkan data berikut.
USER:
- Simbol: {symbol}
- Timeframe: {TIMEFRAME}
- Pemicu Sinyal: Harga telah memasuki zona Fair Value Gap (FVG) / Imbalance.
- Tipe FVG: {fvg['type']}
- Zona FVG: {fvg['min']} - {fvg['max']}
- Harga Saat Ini: {current_price}

Berikan output dalam format JSON dengan kunci "keputusan" (isinya "LONG", "SHORT", atau "WAIT") dan "alasan" (penjelasan singkat).
Contoh: {{"keputusan": "SHORT", "alasan": "Harga mengisi FVG bearish dan kemungkinan akan melanjutkan penurunan."}}"""

                model = genai.GenerativeModel('gemini-1.5-flash-latest')
                response = await model.generate_content_async(prompt)
                
                await bot.send_message(CHAT_ID, f"⚡️ **Sinyal Potensial (Fair Value Gap)** ⚡️\n\n*Simbol:* `{symbol}`\n*Timeframe:* `{TIMEFRAME}`\n*Harga Masuk:* `{current_price}`\n*Zona FVG ({fvg['type']}):* `{fvg['min']} - {fvg['max']}`\n\n*Analisis AI:* \n```{response.text.strip()}```")
                
                alerted[fvg_alert_id] = True

        if not ob and not fvg:
            print(f"[SKIP] Tidak ada OB atau FVG yang ditemukan untuk {symbol}.")

    except Exception as e:
        error_message = f"[FATAL ERROR] Terjadi kesalahan saat menganalisis {symbol}: {e}"
        print(error_message)
        await bot.send_message(CHAT_ID, error_message)

# --- 5. Loop Utama ---

async def main():
    global exchange
    try:
        # Inisialisasi koneksi exchange
        exchange = getattr(ccxt, EXCHANGE_NAME)({'enableRateLimit': True})
        await bot.send_message(CHAT_ID, f"✅ Bot SMC v2.0 (OB/FVG) telah dimulai.\nMemantau {len(SYMBOLS)} simbol pada timeframe {TIMEFRAME}.")
        print(" Bot dimulai...")

        while True:
            print("\n--- Memulai Siklus Analisis Baru ---")
            await asyncio.gather(*[analyze(sym) for sym in SYMBOLS])
            print(f"--- Siklus Selesai. Tidur selama {INTERVAL // 60} menit... ---")
            await asyncio.sleep(INTERVAL)

    finally:
        if exchange:
            await exchange.close()
            print("Koneksi exchange telah ditutup.")
            await bot.send_message(CHAT_ID, "❌ Bot dihentikan. Koneksi exchange ditutup.")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot dihentikan oleh pengguna.")