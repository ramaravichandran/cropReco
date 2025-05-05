from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        features = [
            float(request.form['nitrogen']),
            float(request.form['phosphorus']),
            float(request.form['potassium']),
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['ph']),
            float(request.form['rainfall'])
        ]

        input_data = np.array([features])
        prediction = model.predict(input_data)
        crop = prediction[0]

        # Render template with predicted crop
        return render_template('index.html', crop=crop)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
