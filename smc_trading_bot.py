import asyncio
import ccxt.async_support as ccxt
import nest_asyncio
from telegram import Bot
from telegram.error import TelegramError
import pandas as pd
import numpy as np
import os
import time

nest_asyncio.apply()

# --- KONFIGURASI ---
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'DOGE/USDT',
    'ADA/USDT', 'AVAX/USDT', 'TON/USDT', 'SHIB/USDT', 'LTC/USDT'
]
LOWER_TIMEFRAME = '15m'
HIGHER_TIMEFRAME = '1h'
EXCHANGE_NAME = 'kraken'
DETECTION_MODE = 'FVG'  # atau 'OB'
CANDLE_LIMIT = 100
MIN_IMPULSE_CANDLES = 3
MIN_BODY_PERCENTAGE = 0.5
CHECK_INTERVAL_SECONDS = 300
ATR_PERIOD = 14
BUFFER_ATR_MULTIPLIER = 1.5
RR_RATIO = 2.0

alerted_pois = {}

def calculate_atr(ohlcv, period=14):
    if len(ohlcv) < period:
        return 0.0
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean().iloc[-1]
    return atr if not np.isnan(atr) else 0.0

def find_swing_points(ohlcv, lookback=5):
    if len(ohlcv) < (2 * lookback + 1):
        return [], []
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['is_swing_high'] = (df['high'] == df['high'].rolling(window=2*lookback+1, center=True).max())
    df['is_swing_low'] = (df['low'] == df['low'].rolling(window=2*lookback+1, center=True).min())
    swing_highs = df[df['is_swing_high']][['timestamp', 'high']].to_dict('records')
    swing_lows = df[df['is_swing_low']][['timestamp', 'low']].to_dict('records')
    return swing_highs, swing_lows

def detect_bos_from_swings(ohlcv, lookback=5):
    if len(ohlcv) < (2 * lookback + 2):
        return None
    swing_highs, swing_lows = find_swing_points(ohlcv, lookback=lookback)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None
    recent_swing_high = swing_highs[-2]['high']
    recent_swing_low = swing_lows[-2]['low']
    current_price = ohlcv[-1][4]
    if current_price > recent_swing_high:
        return 'Bullish BOS'
    elif current_price < recent_swing_low:
        return 'Bearish BOS'
    return None

def find_latest_fvg(ohlcv, swing_highs, swing_lows):
    if len(ohlcv) < 3:
        return None
    atr = calculate_atr(ohlcv)
    if atr == 0:
        atr = (ohlcv[-1][2] - ohlcv[-1][3]) or 0.01
    for i in range(len(ohlcv) - 2, 1, -1):
        prev_candle, fvg_candle, next_candle = ohlcv[i-1], ohlcv[i], ohlcv[i+1]
        is_bullish_fvg = fvg_candle[3] > prev_candle[2]
        is_bearish_fvg = fvg_candle[2] < prev_candle[3]
        if is_bullish_fvg:
            return {'type': 'Bullish FVG', 'min_price': prev_candle[2], 'max_price': fvg_candle[3], 'timestamp': fvg_candle[0]}
        if is_bearish_fvg:
            return {'type': 'Bearish FVG', 'min_price': fvg_candle[2], 'max_price': prev_candle[3], 'timestamp': fvg_candle[0]}
    return None

def calculate_sl_tp(poi, poi_type, atr, rr_ratio):
    entry = (poi['min_price'] + poi['max_price']) / 2
    if poi_type.startswith('Bullish'):
        sl = poi['min_price'] - atr * BUFFER_ATR_MULTIPLIER
        tp = entry + abs(entry - sl) * rr_ratio
    else:
        sl = poi['max_price'] + atr * BUFFER_ATR_MULTIPLIER
        tp = entry - abs(entry - sl) * rr_ratio
    return {'entry': entry, 'sl': sl, 'tp': tp}

async def send_telegram_message(bot, message):
    try:
        await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode='Markdown')
        print("[✓] Alert dikirim ke Telegram.")
    except TelegramError as e:
        print(f"[X] Gagal mengirim pesan Telegram: {e}")

async def analyze_symbol(symbol, bot):
    exchange = None
    try:
        exchange = getattr(ccxt, EXCHANGE_NAME)({'enableRateLimit': True})
        print(f"🔍 Menganalisis {symbol}...")

        ohlcv_high = await exchange.fetch_ohlcv(symbol, HIGHER_TIMEFRAME, limit=CANDLE_LIMIT)
        if not ohlcv_high or len(ohlcv_high) < 20:
            print(f"[{symbol}] Data di {HIGHER_TIMEFRAME} tidak cukup.")
            return

        bos = detect_bos_from_swings(ohlcv_high, lookback=5)
        if not bos:
            print(f"[{symbol}] Tidak ada BOS dari swing point.")
            return

        ohlcv_low = await exchange.fetch_ohlcv(symbol, LOWER_TIMEFRAME, limit=CANDLE_LIMIT)
        if not ohlcv_low or len(ohlcv_low) < 20:
            print(f"[{symbol}] Data di {LOWER_TIMEFRAME} tidak cukup.")
            return

        swing_highs, swing_lows = find_swing_points(ohlcv_low)
        current_price = ohlcv_low[-1][4]
        poi = None

        if DETECTION_MODE == 'FVG':
            poi = find_latest_fvg(ohlcv_low, swing_highs, swing_lows)
        elif DETECTION_MODE == 'OB':
            poi = find_latest_order_block(ohlcv_low, MIN_IMPULSE_CANDLES, MIN_BODY_PERCENTAGE, swing_highs, swing_lows)

        if poi and poi['min_price'] <= current_price <= poi['max_price']:
            if symbol not in alerted_pois or alerted_pois.get(symbol) != poi['timestamp']:
                atr = calculate_atr(ohlcv_low, ATR_PERIOD)
                levels = calculate_sl_tp(poi, poi['type'], atr, RR_RATIO)
                rr = round(abs(levels['tp'] - levels['entry']) / abs(levels['entry'] - levels['sl']), 2) if levels['entry'] != levels['sl'] else 'N/A'
                message = (
                    f"🚨 *ALERT ENTRY* 🚨\n\n"
                    f"*{symbol}* memasuki zona POI:\n"
                    f"`{poi['type']}` (konfirmasi `{bos}` di `{HIGHER_TIMEFRAME}`)\n\n"
                    f"💰 Entry: `${levels['entry']:.4f}`\n"
                    f"🛑 SL: `${levels['sl']:.4f}`\n"
                    f"🎯 TP: `${levels['tp']:.4f}`\n"
                    f"---_Harga Saat Ini: `${current_price:.4f}`_---\n"
                    f"📏 RR: `1:{rr}`"
                )
                await send_telegram_message(bot, message)
                alerted_pois[symbol] = poi['timestamp']

    except Exception as e:
        print(f"[X] Error saat menganalisis {symbol}: {e}")
    finally:
        if exchange:
            await exchange.close()

async def main():
    if not all([TELEGRAM_TOKEN, CHAT_ID]):
        print("[X] TELEGRAM_TOKEN dan CHAT_ID belum diatur.")
        return

    bot = Bot(token=TELEGRAM_TOKEN)
    await send_telegram_message(bot, f"✅ Bot Alerter v2.0 Aktif.\nMode: `{DETECTION_MODE}`\nMemantau: `{len(SYMBOLS)}` simbol.")

    while True:
        print(f"\n--- Memulai Siklus Baru: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
        try:
            tasks = [analyze_symbol(symbol, bot) for symbol in SYMBOLS]
            await asyncio.gather(*tasks)
            print(f"--- Selesai. Menunggu {CHECK_INTERVAL_SECONDS} detik... ---")
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)
        except Exception as e:
            print(f"[!!!] Error pada loop utama: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[✓] Bot dihentikan secara manual.")
