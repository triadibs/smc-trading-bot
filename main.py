import os
import asyncio
import ccxt.async_support as ccxt
import nest_asyncio
import json
import pandas as pd
import logging
from telegram import Bot
import google.generativeai as genai

# Menerapkan nest_asyncio
nest_asyncio.apply()

# Konfigurasi Logging
logging.basicConfig(
    filename='/app/trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger().addHandler(logging.StreamHandler())  # Log to console for Fly.io

# Konfigurasi & Inisialisasi
try:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
    CHAT_ID = os.getenv('CHAT_ID')

    missing_vars = [var for var in ['GEMINI_API_KEY', 'TELEGRAM_TOKEN', 'CHAT_ID'] if not os.getenv(var)]
    if missing_vars:
        logging.error(f"Variabel lingkungan berikut belum diatur: {', '.join(missing_vars)}")
        raise ValueError(f"Variabel lingkungan berikut belum diatur: {', '.join(missing_vars)}")

    genai.configure(api_key=GEMINI_API_KEY)
    bot = Bot(token=TELEGRAM_TOKEN)
    exchange = None
except Exception as e:
    logging.error(f"Gagal menginisialisasi layanan: {e}")
    raise

# Pengaturan Strategi
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'DOGE/USDT', 'ADA/USDT']
TIMEFRAME = '15m'
EXCHANGE_NAME = 'kraken'
CANDLES = 200
INTERVAL = 300
ALERTED_MAX_AGE = 86400
alerted = {}

def clean_alerted():
    current_time = pd.Timestamp.now().timestamp() * 1000
    global alerted
    alerted = {k: v for k, v in alerted.items() if current_time - float(k.split('_')[-1]) < ALERTED_MAX_AGE * 1000}
    logging.info("Membersihkan entri alerted yang kadaluarsa.")

def find_order_block(df: pd.DataFrame, periods: int = 5, threshold: float = 0.5, use_wicks: bool = True) -> dict | None:
    try:
        df_ = df.copy()
        ob_period = periods + 1

        for i in range(len(df_) - 1, ob_period, -1):
            window = df_.iloc[i - ob_period : i]
            potential_ob_candle = window.iloc[0]
            subsequent_candles = window.iloc[1:]

            close_ob_potential = potential_ob_candle['c']
            close_last_in_sequence = subsequent_candles.iloc[-1]['c']
            
            if close_ob_potential == 0:
                continue
                
            absmove = ((abs(close_last_in_sequence - close_ob_potential)) / close_ob_potential) * 100
            relmove = absmove >= threshold

            if not relmove:
                continue

            if potential_ob_candle['v'] < df_['v'].mean() * 1.5:
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
    except Exception as e:
        logging.error(f"Error di find_order_block: {e}")
        return None

async def analyze(symbol: str):
    try:
        logging.info(f"Menganalisis {symbol} pada timeframe {TIMEFRAME}")
        ohlcv = await exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=CANDLES)
        if len(ohlcv) < CANDLES:
            logging.warning(f"Data tidak cukup untuk {symbol} ({len(ohlcv)}/{CANDLES} lilin)")
            return

        df = pd.DataFrame(ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])
        current_price = df['c'].iloc[-1]

        ob = find_order_block(df)
        if ob and ob['min'] <= current_price <= ob['max'] and not alerted.get(f"{symbol}_OB_{ob['t']}"):
            logging.info(f"Harga {symbol} ({current_price}) masuk ke Order Block ({ob['type']})")
            trade_data = {
                "asset": symbol,
                "timeframe": TIMEFRAME,
                "trade_direction_potential": "BUY" if ob['type'] == 'Bullish' else "SELL",
                "entry_price": current_price,
                "market_structure": {
                    "current_swing_high": float(df.tail(50)['h'].max()),
                    "current_swing_low": float(df.tail(50)['l'].min())
                },
                "order_block": {
                    "ob_type": ob['type'],
                    "high": ob['max'],
                    "low": ob['min'],
                },
                "primary_target": {
                    "target_price": float(df.tail(50)['h'].max()) if ob['type'] == 'Bullish' else float(df.tail(50)['l'].min()),
                    "target_description": "Previous Swing High" if ob['type'] == 'Bullish' else "Previous Swing Low"
                }
            }

            prompt = f"""
SYSTEM: Anda adalah seorang analis trading Smart Money Concepts (SMC) profesional. Tugas Anda adalah memberikan rekomendasi Stop Loss (SL) dan Take Profit (TP) yang logis berdasarkan data yang diberikan.
USER: Data Setup Trading:
```json
{json.dumps(trade_data, indent=2)}
```
Berikan hasil dalam format JSON dengan field: recommended_sl, recommended_tp, analysis_summary.
"""

            try:
                model = genai.GenerativeModel('gemini-1.5-flash-latest')
                response = model.generate_content(prompt)
                clean_response = response.text.strip().replace("```json", "").replace("```", "")
                if not clean_response:
                    raise ValueError("Respons AI kosong atau tidak valid")
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
                alerted[f"{symbol}_OB_{ob['t']}"] = True
                logging.info(f"Notifikasi untuk {symbol} dikirim: {message}")
            except (json.JSONDecodeError, ValueError) as e:
                logging.error(f"Error parsing AI response untuk {symbol}: {e}\nRaw response: {response.text}")
                await bot.send_message(CHAT_ID, f"⚠️ Gagal mem-parsing respon AI untuk {symbol}.\nRespon mentah:\n`{response.text}`")
        else:
            logging.info(f"Tidak ada Order Block yang valid untuk {symbol}")
    except Exception as e:
        logging.error(f"Gagal menganalisis {symbol}: {e}")

async def main():
    global exchange
    try:
        exchange = ccxt.async_support.kraken({
            'enableRateLimit': True,
        })
        logging.info("Koneksi exchange diinisialisasi")
    except Exception as e:
        logging.error(f"Gagal menginisialisasi exchange: {e}")
        raise

    try:
        await bot.send_message(CHAT_ID, "✅ Bot dimulai.")
        while True:
            current_time_str = pd.Timestamp.now(tz='Asia/Jakarta').strftime('%Y-%m-%d %H:%M:%S')
            logging.info(f"Memulai siklus analisis baru ({current_time_str})")
            print(f"\n--- Memulai Siklus Analisis Baru ({current_time_str}) ---")
            
            clean_alerted()
            await asyncio.gather(*[analyze(sym) for sym in SYMBOLS])
            
            print(f"--- Siklus Selesai. Tidur selama {INTERVAL // 60} menit... ---")
            logging.info(f"Siklus selesai. Menunggu {INTERVAL // 60} menit.")
            await asyncio.sleep(INTERVAL)
    except KeyboardInterrupt:
        logging.info("Bot dihentikan oleh pengguna")
    except Exception as e:
        logging.error(f"Kesalahan fatal di loop utama: {e}")
    finally:
        if exchange:
            await exchange.close()
            logging.info("Koneksi exchange ditutup")
            await bot.send_message(CHAT_ID, "❌ Bot dihentikan. Koneksi exchange ditutup.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"Kesalahan saat menjalankan bot: {e}")