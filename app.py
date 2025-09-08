from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('student_performance_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction_text = None
    if request.method == 'POST':
        try:
            # Get values from the form
            features = [
                float(request.form['age']),
                1 if request.form['gender'] == 'Female' else 0,  # Encode gender
                # Encode ethnicity
                1 if request.form['ethnicity'] == 'Caucasian' else (
                    2 if request.form['ethnicity'] == 'Asian' else 0
                ),
                # Encode parental education
                int({
                    'High School': 0,
                    'Some College': 1,
                    'Bachelor\'s': 2
                }[request.form['parentalEducation']]),
                float(request.form['studyTime']),
                int(request.form['absences']),
                1 if request.form['tutoring'] == 'Yes' else 0,
                # Encode parental support
                int({
                    'Low': 0,
                    'Moderate': 1,
                    'High': 2
                }[request.form['parentalSupport']]),
                1 if request.form['extracurricular'] == 'Yes' else 0,
                1 if request.form['sports'] == 'Yes' else 0,
                1 if request.form['music'] == 'Yes' else 0,
                1 if request.form['volunteering'] == 'Yes' else 0,
                float(request.form['gpa'])
            ]
            
            # Make prediction
            features = np.array(features).reshape(1, -1)
            prediction = model.predict(features)[0]
            
            # Format prediction text
            prediction_text = f"Grade {prediction}"
            
            # Print debug information
            print("Features:", features)
            print("Prediction:", prediction)
            
        except Exception as e:
            print("Error making prediction:", str(e))
            prediction_text = "Error making prediction"
    
    return render_template('index.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True, port=5000)