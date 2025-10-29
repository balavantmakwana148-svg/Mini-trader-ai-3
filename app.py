from fastapi import FastAPI
import yfinance as yf
import joblib
from utils.indicators import add_indicators, detect_candle_pattern
import requests

app = FastAPI()

# Telegram Alert Function
def send_telegram(msg):
    token = "YOUR_TELEGRAM_BOT_TOKEN"
    chat_id = "YOUR_CHAT_ID"
    requests.post(f"https://api.telegram.org/bot{token}/sendMessage", data={"chat_id": chat_id, "text": msg})

# Load Trained Model
model = joblib.load("model/trend_model.pkl")

@app.get("/")
def home():
    return {"message": "Trading AI v2 is Live!"}

@app.get("/signal")
def get_signal(symbol: str = "RELIANCE.NS"):
    df = yf.download(symbol, period="5d", interval="15m", progress=False).dropna()
    df = add_indicators(df)
    pattern = detect_candle_pattern(df)
    X = df[['rsi', 'ema20', 'ema50']].iloc[-1:].values
    prob = model.predict_proba(X)[0][1]
    signal = "BUY" if prob > 0.65 else "SELL" if prob < 0.35 else "HOLD"
    
    message = f"{symbol}: {signal} ({pattern}) - Confidence {prob:.2f}"
    if signal != "HOLD":
        send_telegram(message)
        
    return {
        "symbol": symbol,
        "pattern": pattern,
        "rsi": round(float(df['rsi'].iloc[-1]), 2),
        "signal": signal,
        "confidence": round(float(prob), 2)
}
