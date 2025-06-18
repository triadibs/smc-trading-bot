import asyncio
import ccxt.async_support as ccxt
import nest_asyncio
from telegram import Bot
import pandas as pd
import numpy as np
import os
import time

nest_asyncio.apply()

# --- KONFIGURASI ---
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT']
LOWER_TIMEFRAME = '15m'
HIGHER_TIMEFRAME = '1h'
EXCHANGE_NAME = 'kraken'
CANDLE_LIMIT = 100
CHECK_INTERVAL_SECONDS = 300
ATR_PERIOD = 14
BUFFER_ATR_MULTIPLIER = 1.5
RR_RATIO = 2.0

# Cache BOS and POI state
bos_state = {}  # Format: {symbol: {'type': 'Bullish BOS', 'timestamp': xxx}}
alerted_pois = {}

def calculate_atr(ohlcv, period=14):
    df = pd.DataFrame(ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])
    tr = pd.concat([
        df['h'] - df['l'],
        (df['h'] - df['c'].shift()).abs(),
        (df['l'] - df['c'].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean().iloc[-1]
    return atr if not np.isnan(atr) else 0.0

def find_swing_points(ohlcv, lookback=5):
    df = pd.DataFrame(ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])
    highs = df['h'] == df['h'].rolling(2*lookback+1, center=True).max()
    lows = df['l'] == df['l'].rolling(2*lookback+1, center=True).min()
    return df[highs][['t', 'h']].to_dict('records'), df[lows][['t', 'l']].to_dict('records')

def detect_bos(ohlcv, lookback=5):
    highs, lows = find_swing_points(ohlcv, lookback)
    if len(highs) < 2 or len(lows) < 2: return None
    price = ohlcv[-1][4]
    if price > highs[-2]['h']: return {'type': 'Bullish BOS', 'timestamp': ohlcv[-1][0]}
    if price < lows[-2]['l']: return {'type': 'Bearish BOS', 'timestamp': ohlcv[-1][0]}
    return None

def find_latest_fvg(ohlcv):
    for i in range(len(ohlcv)-2, 1, -1):
        prev, mid, nxt = ohlcv[i-1], ohlcv[i], ohlcv[i+1]
        if mid[3] > prev[2]:
            return {'type': 'Bullish FVG', 'min_price': prev[2], 'max_price': mid[3], 'timestamp': mid[0]}
        if mid[2] < prev[3]:
            return {'type': 'Bearish FVG', 'min_price': mid[2], 'max_price': prev[3], 'timestamp': mid[0]}
    return None

def calculate_sl_tp(poi, poi_type, atr, rr_ratio):
    entry = (poi['min_price'] + poi['max_price']) / 2
    if poi_type.startswith('Bullish'):
        sl = poi['min_price'] - atr * BUFFER_ATR_MULTIPLIER
        tp = entry + (entry - sl) * rr_ratio
    else:
        sl = poi['max_price'] + atr * BUFFER_ATR_MULTIPLIER
        tp = entry - (sl - entry) * rr_ratio
    return {'entry': entry, 'sl': sl, 'tp': tp}

async def send_telegram_message(bot, message):
    try:
        await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode='Markdown')
    except Exception as e:
        print(f"[X] Telegram error: {e}")

async def analyze_symbol(symbol, bot):
    exchange = getattr(ccxt, EXCHANGE_NAME)({'enableRateLimit': True})
    try:
        high_tf = await exchange.fetch_ohlcv(symbol, HIGHER_TIMEFRAME, limit=CANDLE_LIMIT)
        bos = detect_bos(high_tf)
        if bos:
            bos_state[symbol] = bos

        if symbol not in bos_state:
            return

        bos_type = bos_state[symbol]['type']
        low_tf = await exchange.fetch_ohlcv(symbol, LOWER_TIMEFRAME, limit=CANDLE_LIMIT)
        poi = find_latest_fvg(low_tf)
        if not poi:
            return

        # Cek apakah FVG searah dengan BOS
        if bos_type.startswith('Bullish') and not poi['type'].startswith('Bullish'):
            return
        if bos_type.startswith('Bearish') and not poi['type'].startswith('Bearish'):
            return

        price = low_tf[-1][4]
        if poi['min_price'] <= price <= poi['max_price']:
            if symbol not in alerted_pois or alerted_pois[symbol] != poi['timestamp']:
                atr = calculate_atr(low_tf)
                levels = calculate_sl_tp(poi, poi['type'], atr, RR_RATIO)
                rr = round(abs(levels['tp'] - levels['entry']) / abs(levels['entry'] - levels['sl']), 2)
                message = (
                    f"🚨 *ENTRY ALERT* 🚨\n\n"
                    f"*{symbol}* memasuki zona `{poi['type']}`\n"
                    f"Konfirmasi: `{bos_type}` di `{HIGHER_TIMEFRAME}`\n\n"
                    f"💰 Entry: `${levels['entry']:.4f}`\n"
                    f"🛑 SL: `${levels['sl']:.4f}`\n"
                    f"🎯 TP: `${levels['tp']:.4f}`\n"
                    f"Harga Saat Ini: `${price:.4f}`\n"
                    f"RR: `1:{rr}`"
                )
                await send_telegram_message(bot, message)
                alerted_pois[symbol] = poi['timestamp']
    except Exception as e:
        print(f"[X] Error {symbol}: {e}")
    finally:
        await exchange.close()

async def main():
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("[X] Token atau Chat ID belum diset.")
        return

    bot = Bot(token=TELEGRAM_TOKEN)
    await send_telegram_message(bot, f"✅ Bot Aktif. Menunggu harga menyentuh POI setelah BOS...")

    while True:
        print(f"\n=== Cycle @ {time.strftime('%H:%M:%S')} ===")
        tasks = [analyze_symbol(symbol, bot) for symbol in SYMBOLS]
        await asyncio.gather(*tasks)
        print(f"Menunggu {CHECK_INTERVAL_SECONDS}s...\n")
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[✓] Bot dihentikan manual.")
