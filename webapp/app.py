import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, render_template
from keras.models import model_from_json
import numpy as np
import logging
import joblib

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model and scaler during application startup
try:
    # Load model architecture
    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    # Load model weights
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.weights.h5")

    # Load the scaler
    scaler = joblib.load('scaler.pkl')  # Ensure 'scaler.pkl' is in the same directory or provide the correct path

    # Check for scaler feature names (for compatibility)
    scaler_feature_names = getattr(scaler, 'feature_names_in_', None)

    logger.info("Model and scaler loaded successfully!")
except Exception as e:
    logger.error("Error loading model or scaler:", exc_info=True)
    raise e

# Define encoding dictionaries for categorical fields
ENCODINGS = {
    'Age': {'less than 40': 0, '40-49': 1, '50-59': 2, '60 or older': 3},
    'Gender': {'male': 0, 'female': 1},
    'BPLevel': {'low': 0, 'normal': 1, 'high': 2},
    'PhysicallyActive': {'none': 0, 'less than half an hr': 1, 'more than half an hr': 2, 'one hr or more': 3},
    'Stress': {'not at all': 0, 'sometimes': 1, 'very often': 2, 'always': 3},
    'JunkFood': {'occasionally': 0, 'often': 1, 'very often': 2, 'always': 3},
    'UrinationFreq': {'not much': 0, 'quite often': 1},
}

# Default values for fields (used if input is missing or invalid)
REQUIRED_FIELDS = {
    'Age': 'less than 40',
    'Gender': 'male',
    'FamilyDiabetes': 'no',
    'PhysicallyActive': 'none',
    'BMI': 0.0,
    'Smoking': 'no',
    'Alcohol': 'no',
    'Sleep': 0,
    'SoundSleep': 0,
    'RegularMedicine': 'no',
    'JunkFood': 'occasionally',
    'Stress': 'not at all',
    'BPLevel': 'low',
    'Pregnancies': 0,
    'GDiabetes': 'no',
    'UrinationFreq': 'not much'
}

@app.route('/predict', methods=['POST'])
def predict():
    # Predict the probability of diabetes based on input features.
    
    try:
        # Retrieve input data
        input_data = request.json

        # Validate input data
        if not input_data:
            raise ValueError('Input data is missing')

        # If input_data is not a list, convert it to a list with a single element
        if not isinstance(input_data, list):
            input_data = [input_data]

        # Check scaler compatibility
        if scaler_feature_names is not None:
            # Disabled logger.info("Scaler expects feature names: %s", scaler_feature_names)

            # Align input features with scaler's expected order if required
            feature_order = list(scaler_feature_names)
            aligned_X = []
            for record in input_data:
                aligned_X.append(align_features(record, feature_order))
            X = np.array(aligned_X)

        # Prepare input vector for batch processing
        X = []
        for record in input_data:
            X.append(prepare_input_vector(record))

        # Convert X to numpy array
        X = np.array(X)

        # Convert X to DataFrame with correct feature names for compatibility with the scaler
        X_df = pd.DataFrame(X, columns=scaler_feature_names)

        # Scale the data using the pre-loaded scaler
        X_scaled = scaler.transform(X_df)
        
        # Make predictions using the loaded model
        predictions = loaded_model.predict(X_scaled)
        logger.debug("Predictions: %s", predictions)

        # Classify risk levels based on prediction probabilities
        results = [
            {
                'diabetic': ("High Risk - Contact Your GP for Further Evaluation."
                             if p[0] >= 0.5 else
                             "Moderate Risk - Lifestyle Changes and Monitoring Recommended."
                             if p[0] >= 0.3 else
                             "Within Safe Range - Regular Check-ups Advised."),
                'probability': float(p[0])
            }
            for p in predictions
        ]

        return jsonify(results)
    except Exception as e:
        logger.error("An error occurred during prediction:", exc_info=True)
        return jsonify({'error': str(e)}), 500

# Define function to prepare input vector
def prepare_input_vector(input_data):
    X = []
    for field, default_value in REQUIRED_FIELDS.items():
        # Get the value or use the default if not provided
        value = input_data.get(field, default_value)

        try:
            # Perform encoding if applicable
            if field in ENCODINGS:
                value = ENCODINGS[field].get(value.lower(), default_value)
            elif field in ['FamilyDiabetes', 'Smoking', 'Alcohol', 'RegularMedicine', 'GDiabetes']:
                value = 1 if str(value).lower() == 'yes' else 0
            # Ensure numerical fields are floats
            value = float(value)
        except Exception as e:
            logger.warning("Invalid value for field '%s': %s. Using default: %s", field, value, default_value)
            value = float(default_value)
        
        X.append(value)

    logger.debug("Prepared input vector: %s", X)
    return np.array(X)

# Align features according to the scaler's expected feature names
def align_features(input_data, feature_order):
    aligned_record = []
    for feature in feature_order:
        if feature in input_data:
            value = input_data[feature]
        else:
            # Use a default value for missing features
            value = REQUIRED_FIELDS.get(feature, 0)
        aligned_record.append(value)
    return aligned_record

# Render the home page for the web application.
@app.route('/')
def index():
    return render_template('index.html')

# Serve static files such as CSS, JavaScript, or images.
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.logger.setLevel(logging.DEBUG)  # Add this line to enable debug logging
    app.run(debug=True)