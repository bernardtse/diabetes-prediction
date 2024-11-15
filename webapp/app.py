from flask import Flask, request, jsonify, send_from_directory, render_template
from keras.models import model_from_json
import numpy as np
import logging
import json

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model architecture and weights during application startup
try:
    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model_weights.h5")

    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error("Error loading model:", exc_info=True)
    raise e

# Define encoding dictionaries
AGE_ENCODING = {'less than 40': 0, '40-49': 1, '50-59': 2,'60 or older': 3}
GENDER_ENCODING = {'male': 0, 'female': 1}
BP_LEVEL_ENCODING = {'low': 0, 'normal': 1, 'high': 2}
PHYSICALLY_ACTIVE_ENCODING = {'none': 0, 'less than half an hr': 1, 'more than half an hr': 2, 'one hr or more': 3}
STRESS_ENCODING = {'not at all': 0, 'sometimes': 1, 'very often': 2, 'always': 3}
JUNKFOOD_ENCODING = {'occasionally': 0, 'often': 1, 'very often': 2, 'always': 3}
URINATION_FREQ_ENCODING = {'not much': 0, 'quite often': 1}

# Define required fields and their default values
REQUIRED_FIELDS = {
    'age': 'less than 40',
    'gender': 'male',
    'family_diabetes': 'no',
    'physicallyactive': 'none',
    'bmi': 0.0,
    'smoking': 'no',
    'alcohol': 'no',
    'sleep': 0,
    'soundsleep': 0,
    'regularmedicine': 'no',
    'junkfood': 'occasionally',
    'stress': 'not at all',
    'bpLevel': 'low',
    'pregnancies': 0,
    'pdiabetes': 'no',
    'urinationfreq': 'not much'
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        input_data = request.json

        # Validate input data
        if not input_data:
            raise ValueError('Input data is missing')

        # If input_data is not a list, convert it to a list with a single element
        if not isinstance(input_data, list):
            input_data = [input_data]

        # Prepare input vector for batch processing
        X = []
        for _ in range(len(input_data)):
            X.append(prepare_input_vector(input_data[_]))
        
        # Make prediction
        predictions = loaded_model.predict(np.array(X))
        logger.debug("Predictions: %s", predictions)  # Add this line

        # Determine the diabetes status for each prediction
        results = []
        for prediction in predictions:
            diabetic = "High Risk - Contact Your GP for Further Evaluation." if prediction[0] >= 0.5 else "Within Safe Range – Regular Follow-up Advised."
            results.append({'diabetic': diabetic, 'probability': float(prediction[0])})

        return jsonify(results)
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
            value = AGE_ENCODING.get(value.lower(), default_value)
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
    print("X: ", X)
    return np.array(X)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.logger.setLevel(logging.DEBUG)  # Add this line to enable debug logging
    app.run(debug=True)

