import os
import joblib
import numpy as np

from flask import Flask, render_template, request

app = Flask(__name__)

# Model path
model_path = os.path.join(os.getcwd(), 'model', 'rf_model.pkl')

# Check if the model exists before loading it
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Ensure the file exists.")

# Load the trained model
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve input data from the form
        co = float(request.form['CO'])
        pt08_s1_co = float(request.form['PT08.S1(CO)'])
        nmhc_gt = float(request.form['NMHC(GT)'])
        c6h6_gt = float(request.form['C6H6(GT)'])
        pt08_s2_nmhc = float(request.form['PT08.S2(NMHC)'])
        nox_gt = float(request.form['NOx(GT)'])

        # Prepare the input data in the correct shape for the model
        input_data = np.array([co, pt08_s1_co, nmhc_gt, c6h6_gt, pt08_s2_nmhc, nox_gt]).reshape(1, -1)

        # Predict the air quality
        prediction = model.predict(input_data)
        return render_template('result.html', prediction=prediction[0])
    except Exception as e:
        return f"Error during prediction: {e}"

        # Prepare the feature vector for prediction
        features = [[co, pt08_s1_co, nmhc_gt, c6h6_gt, pt08_s2_nmhc,
                     nox_gt, pt08_s3_nox, no2_gt, pt08_s4_no2, t, rh, ah, o3_gt]]
        
        # Make prediction using the model
        prediction = model.predict(features)

        # Return the result
        return render_template('result.html', prediction=prediction[0])

    except Exception as e:
        return f"Error during prediction: {e}"

if __name__ == '__main__':
    app.run(debug=True)
