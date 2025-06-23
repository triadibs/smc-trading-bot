import os
import asyncio
import ccxt.async_support as ccxt
import nest_asyncio
import json
import pandas as pd
from telegram import Bot
import google.generativeai as genai

nest_asyncio.apply()

# --- Load credentials dari environment variables ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
assert all([GEMINI_API_KEY, TELEGRAM_TOKEN, CHAT_ID]), "ENV vars belum lengkap!"

# --- Konfigurasi API ---
genai.configure(api_key=GEMINI_API_KEY)
bot = Bot(token=TELEGRAM_TOKEN)

# --- Konstanta utama ---
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'DOGE/USDT', 'ADA/USDT']
TIMEFRAME = '15m'
EXCHANGE_NAME = 'kraken'
CANDLES = 200
INTERVAL = 300  # 5 menit

# --- State global ---
alerted = {}
exchange = None

# --- Deteksi Order Block ---
def detect_order_blocks(df: pd.DataFrame, periods=5, threshold=0.0, use_wicks=False):
    obs = []
    n = periods + 1
    for i in range(n, len(df)):
        ob_candle = df.iloc[i - n]
        sub_df = df.iloc[i - periods:i]

        if ob_candle['close'] < ob_candle['open'] and all(sub_df['close'] > sub_df['open']):
            move_pct = abs(df.iloc[i - 1]['close'] - ob_candle['close']) / ob_candle['close'] * 100
            if move_pct >= threshold:
                high = ob_candle['high'] if use_wicks else ob_candle['open']
                low = ob_candle['low']
                obs.append({
                    'index': i - n,
                    'type': 'bullish',
                    'high': high,
                    'low': low,
                    'avg': (high + low) / 2,
                    't': df.iloc[i - n]['t']
                })

        if ob_candle['close'] > ob_candle['open'] and all(sub_df['close'] < sub_df['open']):
            move_pct = abs(df.iloc[i - 1]['close'] - ob_candle['close']) / ob_candle['close'] * 100
            if move_pct >= threshold:
                low = ob_candle['low'] if use_wicks else ob_candle['open']
                high = ob_candle['high']
                obs.append({
                    'index': i - n,
                    'type': 'bearish',
                    'high': high,
                    'low': low,
                    'avg': (high + low) / 2,
                    't': df.iloc[i - n]['t']
                })
    return obs

# --- Fungsi analisis utama ---
async def analyze(symbol):
    try:
        print(f"🔍 Menganalisis {symbol}...")
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=CANDLES)
        except Exception as e:
            await bot.send_message(CHAT_ID, f"[ERROR] Fetch OHLCV {symbol}: {e}")
            return

        df = pd.DataFrame(ohlcv, columns=['t', 'open', 'high', 'low', 'close', 'volume'])
        obs = detect_order_blocks(df, periods=5, threshold=0.3, use_wicks=False)

        if not obs:
            print(f"[OB] Tidak ada OB valid untuk {symbol}")
            return

        latest_ob = obs[-1]
        cp = df['close'].iloc[-1]

        if latest_ob['low'] <= cp <= latest_ob['high'] and alerted.get(symbol) != latest_ob['t']:
            print(f"[OB-MATCH] {symbol}: Harga masuk OB! ({cp} ∈ [{latest_ob['low']} - {latest_ob['high']}])")

prompt = f"""
Berikut adalah data sinyal yang saya punya:
- Symbol: {symbol}
- Jenis OB: {latest_ob['type'].upper()}
- Area OB: {latest_ob['low']} - {latest_ob['high']}
- Harga sekarang: {cp}

Apa keputusan kamu (LONG / SHORT / SKIP)? Berikan output JSON seperti ini:

{{ "keputusan": "LONG", "alasan": "..." }}
""".strip()

            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = await model.generate_content_async(prompt)
            ai_response = response.text.strip()

            await bot.send_message(CHAT_ID, f"[LOG] {symbol} Gemini:\n{ai_response}")

            try:
                data = json.loads(ai_response.replace("```json", "").replace("```", "").strip())
                keputusan = data.get("keputusan", "").upper()
                if keputusan in ['LONG', 'SHORT']:
                    msg = f"📈 Gemini SIGNAL {symbol}: {keputusan} (OB: {latest_ob['type']})"
                    await bot.send_message(CHAT_ID, msg)
                    alerted[symbol] = latest_ob['t']
            except Exception as e:
                await bot.send_message(CHAT_ID, f"[ERROR] JSON parse {symbol}: {e}")
        else:
            print(f"[NO ENTRY] Harga belum masuk OB {symbol}")

    except Exception as e:
        print(f"[FATAL] {symbol}: {e}")
        await bot.send_message(CHAT_ID, f"[FATAL] Analyze {symbol}: {e}")

# --- Loop utama ---
async def main():
    global exchange
    exchange = getattr(ccxt, EXCHANGE_NAME)({'enableRateLimit': True})
    await bot.send_message(CHAT_ID, "✅ Bot Gemini Order Block berjalan...")
    print("♻️ Loop analisis dimulai.")

    try:
        while True:
            await asyncio.gather(*[analyze(sym) for sym in SYMBOLS])
            print("✅ Siklus selesai. Tidur 5 menit...\n")
            await asyncio.sleep(INTERVAL)
    finally:
        if exchange:
            await exchange.close()
            print("❌ Exchange connection closed.")
            await bot.send_message(CHAT_ID, "❌ Exchange connection closed.")

if __name__ == '__main__':
    asyncio.run(main())
