from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('../models/rf_no2_model.pkl')

@app.route('/')
def home():
    return "Welcome to the NO2 Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Expecting data in same order of features:
    # ['PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
    #  'NOx(GT)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
    #  'T', 'RH', 'AH']
    
    try:
        input_features = [data[feature] for feature in ['PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
                                                        'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)',
                                                        'PT08.S4(NO2)', 'PT08.S5(O3)',
                                                        'T', 'RH', 'AH']]
        
        prediction = model.predict([input_features])
        return jsonify({'predicted_NO2_GT': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
