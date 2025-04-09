from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open(r'C:\Users\likit\Desktop\1\Box-Office-Revenue-Prediction-Using-Linear-Regression-in-ML\box_office_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')  # Homepage with input form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        budget = float(request.form['budget'])
        marketing_expense = float(request.form['marketing_expense'])
        release_date_factor = float(request.form['release_date_factor'])
        
        # Prepare data for prediction
        features = np.array([[budget, marketing_expense, release_date_factor]])
        prediction = model.predict(features)[0]

        # Return result
        return render_template('result.html', prediction=round(prediction, 2))
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
