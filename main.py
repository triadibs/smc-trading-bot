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
HI_TF = '15m'
LO_TF = '5m'
EXCHANGE_NAME = 'kraken'
CANDLES = 200
INTERVAL = 300  # 5 menit

# --- State global ---
bos_state = {}
alerted = {}
exchange = None  # akan diinisialisasi nanti

# --- Fungsi-fungsi analisis ---
def find_swing(df, lookback=10):
    highs, lows = [], []
    for i in range(lookback, len(df) - lookback):
        window_h = df['h'][i - lookback: i + lookback + 1]
        window_l = df['l'][i - lookback: i + lookback + 1]
        if df['h'][i] >= window_h.max():
            highs.append({'t': df['t'][i], 'p': df['h'][i]})
        if df['l'][i] <= window_l.min():
            lows.append({'t': df['t'][i], 'p': df['l'][i]})
    return highs, lows

def detect_bos(df):
    highs, lows = find_swing(df)
    cp = df['c'].iloc[-1]
    if highs and cp > highs[-1]['p']:
        return {'type': 'Bullish', 'p': highs[-1]['p']}
    if lows and cp < lows[-1]['p']:
        return {'type': 'Bearish', 'p': lows[-1]['p']}
    return None

def find_fvg(ohlcv):
    for i in range(len(ohlcv) - 3, 0, -1):
        c1, c3 = ohlcv[i], ohlcv[i + 2]
        if c3[3] > c1[2]:  # Bullish FVG
            return {'type': 'Bullish FVG', 'min': c1[2], 'max': c3[3], 't': ohlcv[i + 1][0]}
        if c3[2] < c1[3]:  # Bearish FVG
            return {'type': 'Bearish FVG', 'min': c3[2], 'max': c1[3], 't': ohlcv[i + 1][0]}
    return None

async def ask_gemini(symbol, bos, poi, cp):
    prompt = f"""SYSTEM: Kamu analis SMC. Evaluasi data berikut dan berikan keputusan sinyal trading.
USER:
Symbol: {symbol}
BOS: {bos['type']} @ {bos['p']}
POI: {poi['type']} {poi['min']} - {poi['max']}
Harga saat ini: {cp}
Output dalam JSON:
{{"keputusan": "LONG atau SHORT atau WAIT"}}"""

    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = await model.generate_content_async(prompt)
    return response.text.strip()

async def analyze(symbol):
    try:
        print(f"🔍 Menganalisis {symbol}...")
        # --- Timeframe besar ---
        try:
            high_ohlcv = await exchange.fetch_ohlcv(symbol, HI_TF, limit=CANDLES)
        except Exception as e:
            print(f"[ERROR] Fetch OHLCV {symbol} 1h: {e}")
            await bot.send_message(CHAT_ID, f"[ERROR] Fetch OHLCV {symbol} 1h: {e}")
            return

        dfh = pd.DataFrame(high_ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])
        bos = detect_bos(dfh)
        if bos:
            print(f"[BOS] {symbol}: {bos}")
            bos_state[symbol] = bos

        if symbol not in bos_state:
            print(f"[SKIP] Tidak ada BOS untuk {symbol}")
            return

        # --- Timeframe kecil ---
        try:
            low_ohlcv = await exchange.fetch_ohlcv(symbol, LO_TF, limit=CANDLES)
        except Exception as e:
            print(f"[ERROR] Fetch OHLCV {symbol} 15m: {e}")
            await bot.send_message(CHAT_ID, f"[ERROR] Fetch OHLCV {symbol} 15m: {e}")
            return

        poi = find_fvg(low_ohlcv)
        if not poi or not poi['type'].startswith(bos_state[symbol]['type']):
            print(f"[POI] Tidak cocok untuk {symbol}")
            return

        cp = low_ohlcv[-1][4]  # Close price
        if poi['min'] <= cp <= poi['max'] and alerted.get(symbol) != poi['t']:
            print(f"[MATCH] Harga masuk POI: {cp} ∈ ({poi['min']} - {poi['max']})")
            ai_response = await ask_gemini(symbol, bos_state[symbol], poi, cp)

            await bot.send_message(CHAT_ID, f"[LOG] {symbol} Gemini:\n{ai_response}")

            try:
                data = json.loads(ai_response.replace("```json", "").replace("```", "").strip())
                keputusan = data.get("keputusan", "").upper()
                if keputusan in ['LONG', 'SHORT']:
                    msg = f"📈 Gemini SIGNAL {symbol}: {keputusan} (POI: {poi['type']})"
                    await bot.send_message(CHAT_ID, msg)
                    alerted[symbol] = poi['t']
            except Exception as e:
                print(f"[ERROR] JSON parse {symbol}: {e}")
                await bot.send_message(CHAT_ID, f"[ERROR] JSON parse {symbol}: {e}")
        else:
            print(f"[NO ENTRY] Harga belum masuk POI {symbol}")

    except Exception as e:
        print(f"[FATAL] {symbol}: {e}")
        await bot.send_message(CHAT_ID, f"[FATAL] Analyze {symbol}: {e}")

# --- Loop utama ---
async def main():
    global exchange
    exchange = getattr(ccxt, EXCHANGE_NAME)({'enableRateLimit': True})
    await bot.send_message(CHAT_ID, "✅ Bot Gemini SMC berjalan...")
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
