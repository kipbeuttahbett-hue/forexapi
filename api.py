from flask import Flask, request, jsonify
import pandas as pd
import pandas_ta as ta
import joblib
import numpy as np
import os

app = Flask(__name__)

# ==========================================
# 1. LOAD THE MODEL AND FEATURES
# ==========================================
MODEL_PATH = 'gold_decision_tree.pkl'
FEATURES_PATH = 'gold_features.pkl'

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at {MODEL_PATH}")
    exit(1)

if not os.path.exists(FEATURES_PATH):
    print(f"❌ Features not found at {FEATURES_PATH}")
    exit(1)

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)
print("✅ Model loaded successfully")
print(f"✅ Features loaded: {features}")

# ==========================================
# 2. THE PREDICTION ENDPOINT
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        if 'candles' not in data:
            return jsonify({"error": "Missing 'candles' key in request body"}), 400

        # Convert list of dicts to DataFrame
        df = pd.DataFrame(data['candles'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        # Ensure numeric types
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col])

        # ==========================================
        # 3. FEATURE ENGINEERING (MUST MATCH TRAINING)
        # ==========================================
        df['Returns'] = df['Close'].pct_change()
        df['Return_15'] = df['Close'].pct_change(15)
        df['RSI_14'] = ta.rsi(df['Close'], length=14)
        df['ADX_14'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
        df['SMA_14'] = ta.sma(df['Close'], length=14)
        df['SMA_Ratio'] = df['SMA_14'] / df['Close']
        df['SMA_Corr'] = df['Close'].rolling(14).corr(df['SMA_14'])
        df['Volatility_14'] = df['Returns'].rolling(14).std()
        df['Volatility_210'] = df['Returns'].rolling(210).std()
        df['VWAP_14'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
        df['VWAP_Ratio'] = df['VWAP_14'] / df['Close']
        
        # Drop NaNs (created by rolling windows)
        df = df.dropna()
        
        if len(df) < 1:
            return jsonify({"error": "Not enough data after feature engineering"}), 400
            
        # ==========================================
        # 4. PREDICT
        # ==========================================
        latest = df[features].iloc[-1:]
        
        prediction = model.predict(latest)[0]
        proba = model.predict_proba(latest)[0]
        
        action = "BUY" if prediction == 1 else "SELL"
        confidence = float(max(proba))
        
        # ==========================================
        # 5. ADD CONFIDENCE LABEL
        # ==========================================
        if confidence > 0.70:
            label = "HIGH"
        elif confidence > 0.55:
            label = "MEDIUM"
        else:
            label = "LOW"
        
        return jsonify({
            "action": action,
            "confidence": confidence,
            "label": label,
            "prediction": int(prediction),
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================================
# 6. RUN THE APP
# ==========================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
