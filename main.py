import os
import asyncio
import ccxt.async_support as ccxt
import nest_asyncio
import json
import pandas as pd
import numpy as np
from telegram import Bot
import google.generativeai as genai

# Agar asyncio bisa digunakan di notebook
nest_asyncio.apply()

# --- 1. Konfigurasi & Inisialisasi ---

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')

assert all([GEMINI_API_KEY, TELEGRAM_TOKEN, CHAT_ID]), "Variabel ENV (GEMINI_API_KEY, TELEGRAM_TOKEN, CHAT_ID) belum lengkap!"

genai.configure(api_key=GEMINI_API_KEY)
bot = Bot(token=TELEGRAM_TOKEN)
exchange = None  # akan diisi nanti

# --- 2. Pengaturan Strategi ---

SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'DOGE/USDT', 'ADA/USDT']
TIMEFRAME = '1H'
EXCHANGE_NAME = 'kraken'
CANDLES = 200
INTERVAL = 300  # detik (5 menit)

# Tracking notifikasi
alerted = {}

# --- 3. Fungsi Deteksi Order Block ---

def find_order_block(df: pd.DataFrame, periods: int = 5, threshold: float = 0.5, use_wicks: bool = True) -> dict | None:
    df_ = df.copy()
    ob_period = periods + 1
    for i in range(len(df_) - 1, ob_period, -1):
        window = df_.iloc[i - ob_period: i]
        potential_ob_candle = window.iloc[0]
        subsequent_candles = window.iloc[1:]
        close_ob_potential = potential_ob_candle['c']
        if close_ob_potential == 0:
            continue
        close_last_in_sequence = subsequent_candles.iloc[-1]['c']
        absmove = ((abs(close_last_in_sequence - close_ob_potential)) / close_ob_potential) * 100
        relmove = absmove >= threshold
        if not relmove:
            continue
        is_potential_bullish_ob = potential_ob_candle['c'] < potential_ob_candle['o']
        are_subsequent_up = (subsequent_candles['c'] > subsequent_candles['o']).all()
        if is_potential_bullish_ob and are_subsequent_up:
            ob_high = potential_ob_candle['h'] if use_wicks else potential_ob_candle['o']
            ob_low = potential_ob_candle['l']
            return {'type': 'Bullish', 'min': ob_low, 'max': ob_high, 't': potential_ob_candle['t']}
        is_potential_bearish_ob = potential_ob_candle['c'] > potential_ob_candle['o']
        are_subsequent_down = (subsequent_candles['c'] < subsequent_candles['o']).all()
        if is_potential_bearish_ob and are_subsequent_down:
            ob_high = potential_ob_candle['h']
            ob_low = potential_ob_candle['l'] if use_wicks else potential_ob_candle['o']
            return {'type': 'Bearish', 'min': ob_low, 'max': ob_high, 't': potential_ob_candle['t']}
    return None

# --- 4. Fungsi Analisis Inti ---

async def analyze(symbol: str):
    try:
        print(f"🔍 Menganalisis {symbol} pada timeframe {TIMEFRAME}...")
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=CANDLES)
            if len(ohlcv) < CANDLES:
                print(f"[SKIP] Data tidak cukup untuk {symbol} ({len(ohlcv)}/{CANDLES}).")
                return
        except Exception as e:
            print(f"[ERROR] Gagal mengambil data untuk {symbol}: {e}")
            return

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
                        "low": ob['min']
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
          """  model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = await model.generate_content_async(prompt)

            try:
                clean_response = response.text.strip().replace("```json", "").replace("```", "")
                analysis_result = json.loads(clean_response)
                sl = analysis_result.get("recommended_sl", "N/A")
                tp = analysis_result.get("recommended_tp", "N/A")
                summary = analysis_result.get("analysis_summary", "Tidak ada ringkasan.")
                message = (
                    f"🔥 **Sinyal Analisis (Order Block)** 🔥\n\n"
                    f"*Simbol:* `{symbol}`\n"
                    f"*Timeframe:* `{TIMEFRAME}`\n"
                    f"*Harga Masuk:* `{current_price}`\n"
                    f"*Zona OB ({ob['type']}):* `{ob['min']} - {ob['max']}`\n\n"
                    f"**Rekomendasi AI:**\n"
                    f"🛑 **Stop Loss:** `{sl}`\n"
                    f"🎯 **Take Profit:** `{tp}`\n\n"
                    f"📝 *Ringkasan:* {summary}"
                )
                await bot.send_message(CHAT_ID, message, parse_mode='Markdown')
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing AI response: {e}\nRaw response: {response.text}")
                await bot.send_message(CHAT_ID, f"⚠️ Gagal parsing respon AI untuk {symbol}.\nRespon mentah:\n`{response.text}`")

            alerted[ob_alert_id] = True
    else:
        print(f"[SKIP] Tidak ada Order Block yang valid ditemukan untuk {symbol}.")

except Exception as e:
    error_message = f"[FATAL ERROR] Terjadi kesalahan saat menganalisis {symbol}: {e}"
    print(error_message)

#--- 5. Loop Utama ---
async def main():
global exchange
try:
exchange = getattr(ccxt, EXCHANGE_NAME)({'enableRateLimit': True})
await bot.send_message(CHAT_ID, f"✅ Bot Analis vFinal (Order Block Only) telah dimulai.\nMemantau {len(SYMBOLS)} simbol pada timeframe {TIMEFRAME}.")
print("🚀 Bot dimulai...")
    while True:
        current_time_str = pd.Timestamp.now(tz='Asia/Jakarta').strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n--- Memulai Siklus Analisis Baru ({current_time_str}) ---")
        await asyncio.gather(*[analyze(sym) for sym in SYMBOLS])
        print(f"--- Siklus Selesai. Tidur selama {INTERVAL // 60} menit... ---")
        await asyncio.sleep(INTERVAL)

finally:
    if exchange:
        await exchange.close()
        print("🔌 Koneksi exchange telah ditutup.")
        await bot.send_message(CHAT_ID, "❌ Bot dihentikan. Koneksi exchange ditutup.")
#--- 6. Eksekusi Program ---
if name == 'main':
try:
asyncio.run(main())
except KeyboardInterrupt:
print("\nBot dihentikan oleh pengguna.")

