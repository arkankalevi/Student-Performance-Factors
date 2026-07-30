from flask import Flask, request, jsonify
import joblib

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Memuat model yang telah disimpan
joblib_model = joblib.load('lr_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']  # Mengambil data dari request JSON
    prediction = joblib_model.predict([data])[0] 
    return jsonify({'prediction':round(float(prediction), 2)})

if __name__ == '__main__':
    app.run(debug=True)