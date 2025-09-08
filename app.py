from flask import Flask, request, jsonify
import joblib
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline

import os
from  flask_cors import CORS 

cancer_model = joblib.load('logistic_regression_model.pkl')
feature_encoders = joblib.load('feature_encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')


app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Cancer detector API is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])

        categorical_features = ['Grade', 'T Stage', 'N Stage', 'Estrogen Status', 'Progesterone Status']
        for col in categorical_features:
            if col in input_df.columns and col in feature_encoders:
                le = feature_encoders[col]
                input_df[col] = le.transform(input_df[col].astype(str))
            elif col in input_df.columns:
                return jsonify({'error': f'Encoder not found for feature: {col}'}), 400

        predicted_stage_encoded = cancer_model.predict(input_df)
        predicted_stage = target_encoder.inverse_transform(predicted_stage_encoded)

        stage_mapping = {
            "0": "Stage 0 (Very Early Stage, In situ)",
            "I": "Stage I (Early Stage)",
            "IA": "Stage I (Early Stage)",
            "IB": "Stage I (Early Stage)",
            "II": "Stage II (Localized but Larger Tumor)",
            "IIA": "Stage II (Localized but Larger Tumor)",
            "IIB": "Stage II (Localized but Larger Tumor)",
            "III": "Stage III (Advanced, spread to nearby lymph nodes)",
            "IIIA": "Stage III (Advanced, spread to nearby lymph nodes)",
            "IIIB": "Stage III (Advanced, spread to nearby lymph nodes)",
            "IIIC": "Stage III (Advanced, spread to nearby lymph nodes)",
            "IV": "Stage IV (Metastatic, spread to other organs)"
        }
        
        simple_stage = stage_mapping.get(predicted_stage[0], 'Unknown Stage')

       
        return jsonify({
            'predicted_stage': predicted_stage[0],
            'simple_stage': simple_stage,
            
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port)
