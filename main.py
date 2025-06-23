import os
import asyncio
import ccxt.async_support as ccxt
import nest_asyncio
import json
import pandas as pd
from telegram import Bot
from telegram.constants import ParseMode
import google.generativeai as genai

# Mengaplikasikan nest_asyncio agar event loop bisa berjalan di dalam event loop lain.
nest_asyncio.apply()

# --- 1. Load Credentials dan Konfigurasi Awal ---
try:
    GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
    TELEGRAM_TOKEN = os.environ['TELEGRAM_TOKEN']
    CHAT_ID = os.environ['CHAT_ID']
except KeyError as e:
    print(f"FATAL: Variabel environment tidak ditemukan: {e}. Pastikan Anda sudah mengaturnya.")
    exit()

# Konfigurasi API
genai.configure(api_key=GEMINI_API_KEY)
bot = Bot(token=TELEGRAM_TOKEN)

# --- 2. Konstanta Utama ---
BOT_VERSION = "2.1" # Versi bot untuk notifikasi
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'DOGE/USDT', 'ADA/USDT']
TIMEFRAME = '15m'
EXCHANGE_NAME = 'kraken'
CANDLES = 200
INTERVAL = 300  # 5 menit (dalam detik)
ORDER_BLOCK_PERIODS = 5
ORDER_BLOCK_THRESHOLD = 0.3

# --- 3. State Global ---
alerted = {}
exchange = None

# --- 4. Fungsi-fungsi Inti ---

def detect_order_blocks(df: pd.DataFrame, periods=5, threshold=0.0, use_wicks=False):
    """Mendeteksi order block Bullish dan Bearish pada DataFrame OHLCV."""
    obs = []
    n = periods + 1
    if len(df) <= n:
        return obs

    for i in range(n, len(df)):
        ob_candle = df.iloc[i - n]
        sub_df = df.iloc[i - periods:i]

        # Bullish OB
        is_bullish_ob_candidate = ob_candle['close'] < ob_candle['open'] and all(sub_df['close'] > sub_df['open'])
        if is_bullish_ob_candidate:
            move_pct = abs(df.iloc[i - 1]['close'] - ob_candle['close']) / ob_candle['close'] * 100
            if move_pct >= threshold:
                high = ob_candle['high'] if use_wicks else ob_candle['open']
                low = ob_candle['low']
                obs.append({
                    'index': i - n, 'type': 'bullish', 'high': high, 'low': low,
                    'avg': (high + low) / 2, 't': df.iloc[i - n]['t']
                })

        # Bearish OB
        is_bearish_ob_candidate = ob_candle['close'] > ob_candle['open'] and all(sub_df['close'] < sub_df['open'])
        if is_bearish_ob_candidate:
            move_pct = abs(df.iloc[i - 1]['close'] - ob_candle['close']) / ob_candle['close'] * 100
            if move_pct >= threshold:
                low = ob_candle['low'] if use_wicks else ob_candle['open']
                high = ob_candle['high']
                obs.append({
                    'index': i - n, 'type': 'bearish', 'high': high, 'low': low,
                    'avg': (high + low) / 2, 't': df.iloc[i - n]['t']
                })
    return obs

async def analyze(symbol: str):
    """
    Fungsi utama untuk menganalisis satu simbol: mengambil data, mendeteksi OB,
    dan memanggil AI jika harga memasuki zona OB.
    """
    try:
        print(f"🔍 Menganalisis {symbol}...")
        
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=CANDLES)
        except Exception as e:
            print(f"❌ [ERROR] Gagal mengambil OHLCV untuk {symbol}: {e}")
            return

        if not ohlcv or len(ohlcv) < CANDLES:
            print(f"⚠️ Data tidak lengkap untuk {symbol}, hanya ada {len(ohlcv)} candle.")
            return

        df = pd.DataFrame(ohlcv, columns=['t', 'open', 'high', 'low', 'close', 'volume'])
        obs = detect_order_blocks(df, periods=ORDER_BLOCK_PERIODS, threshold=ORDER_BLOCK_THRESHOLD)

        if not obs:
            print(f"ℹ️ Tidak ada Order Block valid yang ditemukan untuk {symbol}")
            return

        latest_ob = obs[-1]
        current_price = df['close'].iloc[-1]

        price_in_zone = latest_ob['low'] <= current_price <= latest_ob['high']
        is_new_alert = alerted.get(symbol) != latest_ob['t']

        if price_in_zone and is_new_alert:
            print(f"🎯 [OB-MATCH] {symbol}: Harga masuk zona OB! ({current_price} ∈ [{latest_ob['low']} - {latest_ob['high']}])")

            try:
                prompt = (
                    f"Anda adalah seorang analis trading ahli. Berdasarkan data berikut, berikan keputusan trading (LONG, SHORT, atau SKIP) dan alasan singkat dalam format JSON.\n"
                    f"Konteks: Order block 'bullish' adalah area support potensial (permintaan), sedangkan order block 'bearish' adalah area resistance potensial (penawaran).\n\n"
                    f"Data Sinyal:\n"
                    f"- Aset Kripto: {symbol}\n"
                    f"- Tipe Order Block Terdeteksi: {latest_ob['type']}\n"
                    f"- Zona Harga Order Block (High-Low): {latest_ob['high']} - {latest_ob['low']}\n"
                    f"- Harga Saat Ini yang Masuk Zona: {current_price}\n\n"
                    f"Berikan jawaban HANYA dalam format JSON seperti ini: {{\"keputusan\": \"LONG/SHORT/SKIP\", \"alasan\": \"Alasan singkat Anda di sini.\"}}"
                )

                model = genai.GenerativeModel('gemini-1.5-flash-latest')
                response = await model.generate_content_async(prompt)
                ai_response = response.text.strip()

                # Log respons AI dalam format code block agar aman
                await bot.send_message(CHAT_ID, f"📝 [LOG] Respons Gemini untuk {symbol}:\n`{ai_response}`", parse_mode=ParseMode.MARKDOWN_V2)

                cleaned_json_str = ai_response.replace("```json", "").replace("```", "").strip()
                data = json.loads(cleaned_json_str)
                
                keputusan = data.get("keputusan", "SKIP").upper()
                alasan = data.get("alasan", "Tidak ada alasan yang diberikan.")

                if keputusan in ['LONG', 'SHORT']:
                    # PERBAIKAN: Hapus format ` ` pada harga dan alasan untuk menghindari error parsing.
                    # Biarkan format * * karena tidak mengandung karakter spesial.
                    msg = (
                        f"🚀 *Sinyal Gemini untuk {symbol}* 🚀\n\n"
                        f"*Keputusan:* {keputusan}\n"
                        f"*Tipe OB:* `{latest_ob['type']}`\n"
                        f"*Harga Saat Ini:* {current_price}\n\n"
                        f"*Alasan:* {alasan}"
                    )
                    await bot.send_message(CHAT_ID, msg, parse_mode=ParseMode.MARKDOWN_V2)
                    alerted[symbol] = latest_ob['t']
                else:
                    print(f"🤖 Gemini memutuskan SKIP untuk {symbol}. Alasan: {alasan}")

            except json.JSONDecodeError as e:
                print(f"❌ [ERROR] Gagal parsing JSON dari Gemini untuk {symbol}: {e}")
                await bot.send_message(CHAT_ID, f"❌ [ERROR] Gagal parsing JSON dari Gemini untuk {symbol}:\n`{ai_response}`", parse_mode=ParseMode.MARKDOWN_V2)
            except Exception as e:
                print(f"❌ [ERROR] Terjadi kesalahan saat interaksi dengan Gemini untuk {symbol}: {e}")
                # Kirim error sebagai teks biasa untuk keamanan
                await bot.send_message(CHAT_ID, f"❌ [ERROR] Terjadi kesalahan saat interaksi dengan Gemini untuk {symbol}: {e}")
        
        elif price_in_zone and not is_new_alert:
            print(f"ℹ️ {symbol}: Harga masih di zona OB, tetapi notifikasi sudah dikirim sebelumnya.")
        else:
            print(f"⏳ {symbol}: Harga ({current_price}) belum memasuki zona OB terbaru ({latest_ob['low']} - {latest_ob['high']}).")

    except Exception as e:
        print(f"💥 [FATAL] Kesalahan tidak terduga pada {symbol}: {e}")

# --- 5. Loop Utama ---

async def main():
    """Loop utama bot yang berjalan terus-menerus."""
    global exchange
    try:
        exchange = getattr(ccxt, EXCHANGE_NAME)({'enableRateLimit': True})
        
        # PERBAIKAN: Escape karakter '.' pada versi bot secara manual.
        safe_bot_version = BOT_VERSION.replace('.', '\\.')
        startup_message = f"✅ *Bot Gemini Order Block v{safe_bot_version} Aktif*\nExchange: `{EXCHANGE_NAME.capitalize()}`"
        await bot.send_message(CHAT_ID, startup_message, parse_mode=ParseMode.MARKDOWN_V2)
        
        print("♻️ Loop analisis dimulai.")

        while True:
            await asyncio.gather(*[analyze(sym) for sym in SYMBOLS])
            print(f"\n✅ Siklus analisis selesai. Bot akan tidur selama {INTERVAL // 60} menit...\n")
            await asyncio.sleep(INTERVAL)

    except Exception as e:
        print(f"🚨 [CRITICAL] Terjadi error di loop utama: {e}")
        # Kirim error kritis dalam format code block agar aman
        await bot.send_message(CHAT_ID, f"🚨 *Bot Berhenti Total*\nTerjadi error kritis di loop utama:\n`{e}`", parse_mode=ParseMode.MARKDOWN_V2)
    finally:
        if exchange:
            await exchange.close()
            print("❌ Koneksi exchange telah ditutup.")
            # Kirim pesan penutupan sebagai teks biasa
            await bot.send_message(CHAT_ID, "🔌 Koneksi ke exchange telah ditutup.")

# --- 6. Titik Masuk Eksekusi Program ---

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:

        print("\n🛑 Bot dihentikan secara manual oleh pengguna. Menutup koneksi...")

