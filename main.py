import os
import asyncio
import ccxt.async_support as ccxt
import nest_asyncio
import json
import pandas as pd
import numpy as np
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
TIMEFRAME = '1H'                # Timeframe yang digunakan
EXCHANGE_NAME = 'kraken'         # Ganti dengan exchange pilihan Anda (kraken, bybit, dll)
CANDLES = 200                    # Jumlah candle yang diambil untuk analisis
INTERVAL = 300                   # Jeda waktu antar siklus analisis dalam detik (300 detik = 5 menit)

# Dictionary untuk melacak notifikasi yang sudah terkirim
alerted = {}

# --- 3. Fungsi Deteksi ---

def find_order_block(df: pd.DataFrame, periods: int = 5, threshold: float = 0.5, use_wicks: bool = True) -> dict | None:
    """
    Mendeteksi Order Block terbaru berdasarkan logika Pine Script:
    1. Satu lilin berlawanan arah.
    2. Diikuti oleh 'periods' lilin searah.
    3. Pergerakan harga melebihi 'threshold' persen.
    """
    df_ = df.copy()
    ob_period = periods + 1

    # Loop dari candle terbaru ke belakang untuk menemukan OB terakhir
    for i in range(len(df_) - 1, ob_period, -1):
        window = df_.iloc[i - ob_period : i]
        
        potential_ob_candle = window.iloc[0]
        subsequent_candles = window.iloc[1:]

        close_ob_potential = potential_ob_candle['c']
        close_last_in_sequence = subsequent_candles.iloc[-1]['c']
        
        if close_ob_potential == 0: continue # Menghindari pembagian dengan nol
            
        absmove = ((abs(close_last_in_sequence - close_ob_potential)) / close_ob_potential) * 100
        relmove = absmove >= threshold

        if not relmove:
            continue

        # Cek OB Bullish
        is_potential_bullish_ob = potential_ob_candle['c'] < potential_ob_candle['o']
        are_subsequent_up = (subsequent_candles['c'] > subsequent_candles['o']).all()

        if is_potential_bullish_ob and are_subsequent_up:
            ob_high = potential_ob_candle['h'] if use_wicks else potential_ob_candle['o']
            ob_low = potential_ob_candle['l']
            return {'type': 'Bullish', 'min': ob_low, 'max': ob_high, 't': potential_ob_candle['t']}

        # Cek OB Bearish
        is_potential_bearish_ob = potential_ob_candle['c'] > potential_ob_candle['o']
        are_subsequent_down = (subsequent_candles['c'] < subsequent_candles['o']).all()

        if is_potential_bearish_ob and are_subsequent_down:
            ob_high = potential_ob_candle['h']
            ob_low = potential_ob_candle['l'] if use_wicks else potential_ob_candle['o']
            return {'type': 'Bearish', 'min': ob_low, 'max': ob_high, 't': potential_ob_candle['t']}
            
    return None

# --- 4. Fungsi Analisis Inti ---

async def analyze(symbol: str):
    """
    Fungsi inti yang menjalankan seluruh alur analisis untuk satu simbol.
    Fokus hanya pada deteksi Order Block.
    """
    try:
        print(f"🔍 Menganalisis {symbol} pada timeframe {TIMEFRAME}...")

        ohlcv = await exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=CANDLES)
        df = pd.DataFrame(ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])
        current_price = df['c'].iloc[-1]

        ob = find_order_block(df)
        
        if ob:
            ob_alert_id = f"{symbol}_OB_{ob['t']}"

            if (ob['min'] <= current_price <= ob['max']) and not alerted.get(ob_alert_id):
                print(f"🔥 [MATCH OB] Harga {symbol} ({current_price}) masuk ke Order Block.")

                recent_candles = df.tail(50)
                trade_data = {
                    "asset": symbol,
                    "timeframe": TIMEFRAME,
                    "trade_direction_potential": "BUY" if ob['type'] == 'Bullish' else "SELL",
                    "entry_price": current_price,
                    "market_structure": {
                        "current_swing_high": float(recent_candles['h'].max()),
                        "current_swing_low": float(recent_candles['l'].min())
                    },
                    "order_block": {
                        "ob_type": ob['type'],
                        "high": ob['max'],
                        "low": ob['min'],
                    },
                    "primary_target": {
                        "target_price": float(recent_candles['h'].max()) if ob['type'] == 'Bullish' else float(recent_candles['l'].min()),
                        "target_description": "Previous Swing High" if ob['type'] == 'Bullish' else "Previous Swing Low"
                    }
                }
                
                prompt = f"""
SYSTEM: Anda adalah seorang analis trading Smart Money Concepts (SMC) profesional. Tugas Anda adalah memberikan rekomendasi Stop Loss (SL) dan Take Profit (TP) yang logis berdasarkan data yang diberikan.

USER:
Data Setup Trading:
```json
{json.dumps(trade_data, indent=2)}