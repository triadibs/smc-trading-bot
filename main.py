import os
import asyncio
import ccxt.async_support as ccxt
import nest_asyncio
import json
import pandas as pd
from telegram import Bot
import google.generativeai as genai

# Mengaplikasikan nest_asyncio untuk lingkungan seperti Jupyter/Spyder
# Jika dijalankan sebagai script .py murni, ini tidak selalu diperlukan tetapi tidak berbahaya.
nest_asyncio.apply()

# --- Load credentials dari environment variables ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
assert all([GEMINI_API_KEY, TELEGRAM_TOKEN, CHAT_ID]), "Variabel environment (GEMINI_API_KEY, TELEGRAM_TOKEN, CHAT_ID) belum lengkap!"

# --- Konfigurasi API ---
genai.configure(api_key=GEMINI_API_KEY)
bot = Bot(token=TELEGRAM_TOKEN)

# --- Konstanta utama ---
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'DOGE/USDT', 'ADA/USDT']
TIMEFRAME = '15m'
EXCHANGE_NAME = 'kraken'
CANDLES = 200
INTERVAL = 300  # 5 menit (dalam detik)

# --- State global ---
alerted = {}
exchange = None

# --- Deteksi Order Block ---
def detect_order_blocks(df: pd.DataFrame, periods=5, threshold=0.0, use_wicks=False):
    obs = []
    n = periods + 1
    if len(df) <= n:
        return obs # Tidak cukup data untuk dianalisis

    for i in range(n, len(df)):
        ob_candle = df.iloc[i - n]
        sub_df = df.iloc[i - periods:i]

        # Bullish OB: 1 candle turun diikuti oleh 'periods' candle naik
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

        # Bearish OB: 1 candle naik diikuti oleh 'periods' candle turun
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
            await bot.send_message(CHAT_ID, f"[ERROR] Gagal mengambil OHLCV untuk {symbol}: {e}")
            return

        if len(ohlcv) < CANDLES:
            print(f"⚠️ Data tidak lengkap untuk {symbol}, hanya ada {len(ohlcv)} candle.")
            return

        df = pd.DataFrame(ohlcv, columns=['t', 'open', 'high', 'low', 'close', 'volume'])
        obs = detect_order_blocks(df, periods=5, threshold=0.3, use_wicks=False)

        if not obs:
            print(f"ℹ️ Tidak ada Order Block valid yang ditemukan untuk {symbol}")
            return

        latest_ob = obs[-1]
        cp = df['close'].iloc[-1]

        # Cek jika harga saat ini masuk ke zona OB terbaru & belum pernah ada notifikasi untuk OB ini
        if latest_ob['low'] <= cp <= latest_ob['high'] and alerted.get(symbol) != latest_ob['t']:
            print(f"🎯 [OB-MATCH] {symbol}: Harga masuk zona OB! ({cp} ∈ [{latest_ob['low']} - {latest_ob['high']}])")

            # <-- AWAL BLOK YANG DIPERBAIKI: Semua kode di bawah ini diberi indentasi agar masuk ke dalam blok 'if'
            prompt = (
                f"Analisis Sinyal Trading Crypto:\n\n"
                f"Anda adalah seorang analis trading ahli. Berdasarkan data berikut, berikan keputusan trading (LONG, SHORT, atau SKIP) dalam format JSON.\n"
                f"Order block bullish adalah area support potensial, sedangkan order block bearish adalah area resistance potensial.\n\n"
                f"Data Sinyal:\n"
                f"- Aset Kripto: {symbol}\n"
                f"- Tipe Order Block Terdeteksi: {latest_ob['type']}\n"
                f"- Zona Harga Order Block (High-Low): {latest_ob['high']} - {latest_ob['low']}\n"
                f"- Harga Saat Ini: {cp}\n\n"
                f"Berikan jawaban hanya dalam format JSON seperti ini: {{\"keputusan\": \"LONG/SHORT/SKIP\", \"alasan\": \"Alasan singkat Anda di sini.\"}}"
            )

            try:
                model = genai.GenerativeModel('gemini-1.5-flash-latest')
                response = await model.generate_content_async(prompt)
                ai_response = response.text.strip()

                await bot.send_message(CHAT_ID, f"[LOG] Respons Gemini untuk {symbol}:\n{ai_response}")

                # Membersihkan respons dari markdown code block
                cleaned_json_str = ai_response.replace("```json", "").replace("```", "").strip()
                data = json.loads(cleaned_json_str)
                
                keputusan = data.get("keputusan", "").upper()
                alasan = data.get("alasan", "Tidak ada alasan.")

                if keputusan in ['LONG', 'SHORT']:
                    msg = f"🚀 Sinyal Gemini untuk {symbol} 🚀\n\nKeputusan: {keputusan}\nTipe OB: {latest_ob['type']}\nHarga Saat Ini: {cp}\n\nAlasan: {alasan}"
                    await bot.send_message(CHAT_ID, msg)
                    alerted[symbol] = latest_ob['t'] # Tandai bahwa notifikasi untuk OB ini telah dikirim
                else:
                    print(f"🤖 Gemini memutuskan SKIP untuk {symbol}. Alasan: {alasan}")

            except json.JSONDecodeError as e:
                await bot.send_message(CHAT_ID, f"[ERROR] Gagal mem-parsing JSON dari Gemini untuk {symbol}: {e}\nRespons asli: {ai_response}")
            except Exception as e:
                await bot.send_message(CHAT_ID, f"[ERROR] Terjadi kesalahan saat berinteraksi dengan Gemini untuk {symbol}: {e}")
            # <-- AKHIR BLOK YANG DIPERBAIKI
            
        else:
            if alerted.get(symbol) == latest_ob['t']:
                 print(f"ℹ️ {symbol}: Harga masih di zona OB, tetapi notifikasi sudah dikirim.")
            else:
                 print(f"⏳ {symbol}: Harga ({cp}) belum memasuki zona OB terbaru ({latest_ob['low']} - {latest_ob['high']}).")

    except Exception as e:
        print(f"[FATAL] Kesalahan tidak terduga pada {symbol}: {e}")
        # await bot.send_message(CHAT_ID, f"[FATAL] Kesalahan pada fungsi analyze untuk {symbol}: {e}")

# --- Loop utama ---
async def main():
    global exchange
    try:
        exchange = getattr(ccxt, EXCHANGE_NAME)({'enableRateLimit': True})
        await bot.send_message(CHAT_ID, f"✅ Bot Gemini Order Block v1.1 telah aktif dan berjalan pada exchange {EXCHANGE_NAME.capitalize()}.")
        print("♻️ Loop analisis dimulai.")

        while True:
            await asyncio.gather(*[analyze(sym) for sym in SYMBOLS])
            print(f"\n✅ Siklus analisis selesai. Bot akan tidur selama {INTERVAL // 60} menit...\n")
            await asyncio.sleep(INTERVAL)

    except Exception as e:
        print(f"[CRITICAL] Terjadi error di loop utama: {e}")
        await bot.send_message(CHAT_ID, f"❌ Bot berhenti karena error kritis di loop utama: {e}")
    finally:
        if exchange:
            await exchange.close()
            print("❌ Koneksi exchange telah ditutup.")
            await bot.send_message(CHAT_ID, "❌ Koneksi ke exchange ditutup.")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Bot dihentikan secara manual oleh pengguna.")
