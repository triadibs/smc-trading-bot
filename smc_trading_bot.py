import asyncio
import ccxt.async_support as ccxt
import nest_asyncio
from telegram import Bot
from telegram.error import TelegramError
import pandas as pd
import numpy as np
import os 

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
CHECK_INTERVAL_SECONDS = 30
ATR_PERIOD = 14
BUFFER_ATR_MULTIPLIER = 1.5  # Buffer SL berdasarkan ATR
RR_RATIO = 2.0

alerted_pois = {}

def calculate_atr(ohlcv, period=14):
    """Menghitung Average True Range (ATR) untuk parameter dinamis."""
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean().iloc[-1]
    return atr if not np.isnan(atr) else 0.0

def detect_break_of_structure(ohlcv):
    """Mendeteksi Break of Structure (BOS) untuk konfirmasi tren."""
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    last_high = df['high'][:-1].max()
    last_low = df['low'][:-1].min()
    current_candle = df.iloc[-1]
    
    if current_candle['close'] > last_high:
        return 'Bullish BOS'
    elif current_candle['close'] < last_low:
        return 'Bearish BOS'
    return None

def find_swing_points(ohlcv, lookback=5):
    """Mendeteksi swing high/low untuk filter likuiditas."""
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    swing_high = df['high'].rolling(window=2*lookback+1, center=True).max()
    swing_low = df['low'].rolling(window=2*lookback+1, center=True).min()
    is_swing_high = (df['high'] == swing_high) & (df['high'] > df['high'].shift(-1)) & (df['high'] > df['high'].shift(1))
    is_swing_low = (df['low'] == swing_low) & (df['low'] < df['low'].shift(-1)) & (df['low'] < df['low'].shift(1))
    swing_highs = df[is_swing_high][['timestamp', 'high']].to_dict('records')
    swing_lows = df[is_swing_low][['timestamp', 'low']].to_dict('records')
    return swing_highs, swing_lows

def find_latest_fvg(ohlcv, swing_highs, swing_lows):
    """Mendeteksi Fair Value Gap (FVG) dengan filter likuiditas."""
    atr = calculate_atr(ohlcv)
    for i in range(len(ohlcv) - 3, 0, -1):
        prev = ohlcv[i-1]
        curr = ohlcv[i]
        nxt = ohlcv[i+1]
        
        nearby_swing_high = any(abs(sh['high'] - curr[3]) < atr for sh in swing_highs)
        nearby_swing_low = any(abs(sl['low'] - curr[2]) < atr for sl in swing_lows)

        if curr[3] > prev[2] and (nearby_swing_high or nearby_swing_low):
            return {'type': 'Bullish FVG', 'min_price': prev[2], 'max_price': curr[3], 'timestamp': curr[0]}
        if curr[2] < prev[3] and (nearby_swing_high or nearby_swing_low):
            return {'type': 'Bearish FVG', 'min_price': curr[2], 'max_price': prev[3], 'timestamp': curr[0]}
    return None

def find_latest_order_block(ohlcv, min_candles=3, min_body=0.5, swing_highs=None, swing_lows=None):
    """Mendeteksi Order Block (OB) dengan filter likuiditas."""
    atr = calculate_atr(ohlcv)
    for i in range(len(ohlcv) - (min_candles + 1), 0, -1):
        ob_candle = ohlcv[i]
        nearby_swing_high = any(abs(sh['high'] - ob_candle[2]) < atr for sh in swing_highs)
        nearby_swing_low = any(abs(sl['low'] - ob_candle[3]) < atr for sl in swing_lows)
        
        if ob_candle[4] < ob_candle[1]:
            impulse = all(
                (ohlcv[i+j][4] > ohlcv[i+j][1]) and
                ((abs(ohlcv[i+j][4] - ohlcv[i+j][1]) / (ohlcv[i+j][2] - ohlcv[i+j][3])) >= min_body)
                for j in range(1, min_candles + 1)
            )
            if impulse and (nearby_swing_high or nearby_swing_low):
                return {'type': 'Bullish OB', 'min_price': ob_candle[3], 'max_price': ob_candle[2], 'timestamp': ob_candle[0]}
        elif ob_candle[4] > ob_candle[1]:
            impulse = all(
                (ohlcv[i+j][4] < ohlcv[i+j][1]) and
                ((abs(ohlcv[i+j][4] - ohlcv[i+j][1]) / (ohlcv[i+j][2] - ohlcv[i+j][3])) >= min_body)
                for j in range(1, min_candles + 1)
            )
            if impulse and (nearby_swing_high or nearby_swing_low):
                return {'type': 'Bearish OB', 'min_price': ob_candle[3], 'max_price': ob_candle[2], 'timestamp': ob_candle[0]}
    return None

def calculate_sl_tp(poi, poi_type, atr, rr_ratio=2.0):
    """Menghitung SL dan TP berdasarkan ATR untuk parameter dinamis."""
    min_price = poi['min_price']
    max_price = poi['max_price']
    entry = (min_price + max_price) / 2
    range_poi = abs(max_price - min_price)
    buffer = atr * BUFFER_ATR_MULTIPLIER

    if poi_type.startswith("Bullish"):
        sl = min_price - buffer
        tp = entry + (entry - sl) * rr_ratio
    else:
        sl = max_price + buffer
        tp = entry - (sl - entry) * rr_ratio

    return {
        'entry': entry,
        'sl': sl,
        'tp': tp
    }

async def send_telegram_message(bot, message):
    """Mengirim pesan ke Telegram dengan error handling."""
    try:
        await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode='Markdown')
        print("[✓] Alert dikirim ke Telegram.")
    except TelegramError as e:
        print(f"[X] Gagal mengirim pesan Telegram: {e}")

async def monitor_symbol(symbol, exchange_class, bot):
    global alerted_pois
    exchange = exchange_class({'enableRateLimit': True})
    print(f"🔍 Memulai {symbol}")
    await send_telegram_message(bot, f"📈 Monitoring aktif untuk {symbol}")

    retries = 3
    try:
        while True:
            try:
                ohlcv_low = await exchange.fetch_ohlcv(symbol, LOWER_TIMEFRAME, limit=CANDLE_LIMIT)
                ohlcv_high = await exchange.fetch_ohlcv(symbol, HIGHER_TIMEFRAME, limit=CANDLE_LIMIT)

                if not ohlcv_low or not ohlcv_high:
                    print(f"[{symbol}] Data OHLCV tidak valid.")
                    await asyncio.sleep(CHECK_INTERVAL_SECONDS)
                    continue

                atr = calculate_atr(ohlcv_low, ATR_PERIOD)
                bos = detect_break_of_structure(ohlcv_high)
                if not bos:
                    print(f"[{symbol}] Tidak ada Break of Structure di {HIGHER_TIMEFRAME}. Menunggu...")
                    await asyncio.sleep(CHECK_INTERVAL_SECONDS)
                    continue

                swing_highs, swing_lows = find_swing_points(ohlcv_low)
                current_price = ohlcv_low[-1][4]
                poi = None
                if DETECTION_MODE == 'FVG':
                    poi = find_latest_fvg(ohlcv_low, swing_highs, swing_lows)
                elif DETECTION_MODE == 'OB':
                    poi = find_latest_order_block(ohlcv_low, MIN_IMPULSE_CANDLES, MIN_BODY_PERCENTAGE, swing_highs, swing_lows)

                if poi and poi['min_price'] <= current_price <= poi['max_price']:
                    if symbol not in alerted_pois or alerted_pois[symbol]['timestamp'] != poi['timestamp']:
                        levels = calculate_sl_tp(poi, poi['type'], atr, RR_RATIO)
                        rr = round((levels['tp'] - levels['entry']) / (levels['entry'] - levels['sl']), 2) if poi['type'].startswith('Bullish') else round((levels['entry'] - levels['tp']) / (levels['sl'] - levels['entry']), 2)
                        message = (
                            f"🚨 *ALERT ENTRY* 🚨\n\n"
                            f"{symbol} memasuki zona POI:\n"
                            f"*{poi['type']}* (konfirmasi {bos} di {HIGHER_TIMEFRAME})\n\n"
                            f"💰 Entry: *${levels['entry']:.2f}*\n"
                            f"🛑 SL: *${levels['sl']:.2f}*\n"
                            f"🎯 TP: *${levels['tp']:.2f}*\n"
                            f"📉 Harga Saat Ini: *${current_price:.2f}*\n"
                            f"📏 RR: *1:{rr}*"
                        )
                        await send_telegram_message(bot, message)
                        alerted_pois[symbol] = poi
                    else:
                        print(f"[{symbol}]: Harga ${current_price:.2f} sudah di-alert untuk POI ini.")
                else:
                    print(f"[{symbol}]: Harga ${current_price:.2f} tidak dalam zona POI atau tidak ada POI terdeteksi.")

                await asyncio.sleep(CHECK_INTERVAL_SECONDS)
            except Exception as e:
                print(f"[{symbol}] Error in iteration: {e}")
                retries -= 1
                if retries == 0:
                    print(f"[{symbol}] Gagal setelah 3 percobaan. Menghentikan monitoring.")
                    break
                await asyncio.sleep(CHECK_INTERVAL_SECONDS)
    finally:
        await exchange.close()
        print(f"[{symbol}] Koneksi ke exchange ditutup.")

async def main():
    bot = Bot(token=TELEGRAM_TOKEN)
    exchange_class = getattr(ccxt, EXCHANGE_NAME)
    tasks = [monitor_symbol(symbol, exchange_class, bot) for symbol in SYMBOLS]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[✓] Bot dihentikan.")
