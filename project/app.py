from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import model_from_json

# Load the model architecture from JSON file
with open('model.json', 'r') as json_file:
    model_json = json_file.read()
loaded_model = model_from_json(model_json)

# Load the model weights
loaded_model.load_weights('model.h5')

# Compile the model (if required)
loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from form
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        # Preprocess input data if required
        input_data = np.array([[feature1, feature2]])  # Example input format, adjust as needed
        # Make prediction using the model
        prediction = loaded_model.predict(input_data)
        # Format prediction result as needed
        result = prediction[0][0]  # Example result, adjust as needed
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
