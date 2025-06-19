import os, asyncio, ccxt.async_support as ccxt, nest_asyncio
from telegram import Bot
import pandas as pd, json
import google.generativeai as genai

nest_asyncio.apply()

# Load credentials dari ENV
GEN_API_KEY = os.getenv('GEMINI_API_KEY')
TELE_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
assert all([GEN_API_KEY, TELE_TOKEN, CHAT_ID]), "ENV vars belum lengkap!"

genai.configure(api_key=GEN_API_KEY)
bot = Bot(token=TELE_TOKEN)

SYMBOLS = ['XBT/USD', 'ETH/USD', 'XRP/USD']
HI_TF = '1h'; LO_TF = '15m'
EXCHANGE = 'kraken'
CANDLES = 200; INTERVAL = 300

bos_state = {}; alerted = {}

def find_swing(df, lookback=10):
    highs=[]; lows=[]
    for i in range(lookback, len(df)-lookback):
        if df['h'][i]>=df['h'][i-lookback:i+lookback+1].max(): highs.append({'t':df['t'][i],'p':df['h'][i]})
        if df['l'][i]<=df['l'][i-lookback:i+lookback+1].min(): lows.append({'t':df['t'][i],'p':df['l'][i]})
    return highs, lows

def detect_bos(df):
    highs,lows=find_swing(df); cp=df['c'].iloc[-1]
    if highs and cp>highs[-1]['p']: return {'type':'Bullish','p':highs[-1]['p']}
    if lows and cp<lows[-1]['p']: return {'type':'Bearish','p':lows[-1]['p']}
    return None

def find_fvg(ohlcv):
    for i in range(len(ohlcv)-3,0,-1):
        c1, c3 = ohlcv[i], ohlcv[i+2]
        if c3[3] > c1[2]: return {'type':'Bullish FVG','min':c1[2],'max':c3[3],'t':ohlcv[i+1][0]}
        if c3[2] < c1[3]: return {'type':'Bearish FVG','min':c3[2],'max':c1[3],'t':ohlcv[i+1][0]}
    return None

async def ask_gemini(symbol, bos, poi, cp):
    prompt=f"""SYSTEM: Kamu analis SMC...
USER:
Symbol: {symbol}
BOS: {bos['type']} @ {bos['p']}
POI: {poi['type']} {poi['min']}-{poi['max']}
Harga saat ini: {cp}
Output JSON:
{{"keputusan":"..."}}"""
    model=genai.GenerativeModel('gemini-1.5-flash-latest')
    resp=await model.generate_content_async(prompt)
    return resp.text

async def analyze(sym):
    ex=getattr(ccxt, EXCHANGE)({'enableRateLimit':True})
    high=await ex.fetch_ohlcv(sym, HI_TF, limit=CANDLES)
    dfh=pd.DataFrame(high, columns=['t','o','h','l','c','v'])
    bos=detect_bos(dfh); 
    if bos: bos_state[sym]=bos

    if sym not in bos_state: return
    poi=find_fvg(await ex.fetch_ohlcv(sym, LO_TF, limit=CANDLES))
    if not poi: return
    if not poi['type'].startswith(bos_state[sym]['type']): return

    low=await ex.fetch_ohlcv(sym, LO_TF, limit=CANDLES)
    cp=low[-1][4]
    if poi['min']<=cp<=poi['max'] and alerted.get(sym)!=poi['t']:
        ai=await ask_gemini(sym, bos_state[sym], poi, cp)
        try:
            data=json.loads(ai.strip().replace('```',''))
            if data['keputusan'] in ['LONG','SHORT']:
                msg=f"Gemini SIGNAL {sym}: {data['keputusan']}, SL/TP rekomendasi."
                await bot.send_message(chat_id=CHAT_ID, text=msg)
                alerted[sym]=poi['t']
        except:
            pass
    await ex.close()

async def main():
    await bot.send_message(CHAT_ID, "Bot Gemini SMC berjalan!")
    while True:
        await asyncio.gather(*[analyze(s) for s in SYMBOLS])
        await asyncio.sleep(INTERVAL)

if __name__=='__main__':
    asyncio.run(main())
