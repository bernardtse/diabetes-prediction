from flask import Flask, request, jsonify, send_from_directory, render_template
from keras.models import model_from_json
import numpy as np
import logging
import json

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model architecture and weights
try:
    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model_weights.h5")

    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error("Error loading model:", exc_info=True)

# Define gender encoding dictionary
GENDER_ENCODING = {'male': 0, 'female': 1}

# Define blood pressure level encoding dictionary
BP_LEVEL_ENCODING = {'low': 0, 'normal': 1, 'high': 2}

# Define physically active encoding dictionary
PHYSICALLY_ACTIVE_ENCODING = {'none': 0, 'less than half an hr': 1, 'more than half an hr': 2, 'one hr or more': 3}

# Define stress level encoding dictionary
STRESS_ENCODING = {'not at all': 0, 'sometimes': 1, 'very often': 2, 'always': 3}

# Define junk food consumption encoding dictionary
JUNKFOOD_ENCODING = {'occasionally': 0, 'often': 1, 'very often': 2, 'always': 3}

# Define urination frequency encoding dictionary
URINATION_FREQ_ENCODING = {'not much': 0, 'quite often': 1}

# Define required fields and their default values
REQUIRED_FIELDS = {
    'age': '0-39',
    'gender': 'male',
    'family_diabetes': 0,
    'physicallyactive': 'none',
    'bmi': 0.0,
    'smoking': 0,
    'alcohol': 0,
    'sleep': 0,
    'soundsleep': 0,
    'regularmedicine': 0,
    'junkfood': 'occasionally',
    'stress': 'not at all',
    'bpLevel': 'low',
    'pregancies': 0,
    'pdiabetes': 0,  # corrected field name here
    'urinationfreq': 'not much'
}

# Your existing Flask route code
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        input_data = request.json

        # Validate input data
        if not input_data:
            raise ValueError('Input data is missing')

        # Check if all required fields are present
        missing_fields = [field for field in REQUIRED_FIELDS.keys() if field not in input_data]
        if missing_fields:
            raise ValueError(f'Missing input fields: {", ".join(missing_fields)}')

        # Prepare input vector
        X = prepare_input_vector(input_data)

        # Make prediction
        prediction = loaded_model.predict(X)

        # Determine the diabetes status based on the prediction
        diabetic = "Diabetic" if prediction[0][0] >= 1.5 else "Non-Diabetic"

        # Prepare the response
        response = {
            'diabetic': diabetic,
            'probability': float(prediction[0][0])  
        }
        return jsonify(response)
    except Exception as e:
        logger.error("An error occurred during prediction:", exc_info=True)
        return jsonify({'error': str(e)}), 500

# Define function to prepare input vector
def prepare_input_vector(input_data):
    X = []
    for field, default_value in REQUIRED_FIELDS.items():
        value = input_data.get(field, default_value)
        if not value:
            logger.warning("Missing value for field: %s" % field)
            value = default_value
        if field in ['family_diabetes', 'smoking', 'alcohol', 'regularmedicine', 'pdiabetes']:
            value = 1 if str(value).lower() == 'yes' else 0
        elif field == 'age':
            logger.info("Age value received: %s" % value)
            if value == 'less than 40':
                value = 20
            elif value == '40-49':
                value = (40 + 49) / 2
            elif value == '50-59':
                value = (50 + 59) / 2
            elif value == '60 or older':
                value = 70
            else:
                try:
                    lower_bound, upper_bound = map(int, value.split('-'))
                    value = (lower_bound + upper_bound) / 2
                except ValueError:
                    logger.warning("Invalid age value: %s" % value)
                    value = default_value
        elif field == 'gender':
            value = GENDER_ENCODING.get(value.lower(), default_value)
        elif field == 'bpLevel':
            value = BP_LEVEL_ENCODING.get(value.lower(), default_value)
        elif field == 'physicallyactive':
            value = PHYSICALLY_ACTIVE_ENCODING.get(value.lower(), default_value)
        elif field == 'junkfood':
            value = JUNKFOOD_ENCODING.get(value.lower(), default_value)
        elif field == 'stress':
            value = STRESS_ENCODING.get(value.lower(), default_value)
        elif field == 'urinationfreq':
            value = URINATION_FREQ_ENCODING.get(value.lower(), default_value)

        X.append(float(value))
    
    return np.array([X])

# Serve the index.html file
@app.route('/')
def index():
    return render_template('index.html')

# Serve static files (like script.js and styles.css)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
