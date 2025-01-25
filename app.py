from flask import Flask, request, render_template, redirect, url_for
import json
import os
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Path to store JSON file
json_file_path = 'data/contact_form_data.json'
os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/features.html')
def features():
    return render_template('features.html')

@app.route('/contact.html', methods=['GET', 'POST'])
def contact():
    if request.method == 'GET':
        return render_template('contact.html')
    else:
        try:
            # Collect form data
            name = request.form.get('name')
            email = request.form.get('email')
            message = request.form.get('message')

            form_data = {
                "name": name,
                "email": email,
                "message": message
            }

            # Ensure the JSON file exists and read its content
            if not os.path.exists(json_file_path):
                with open(json_file_path, 'w') as f:
                    json.dump([], f)  # Initialize the file with an empty list

            # Read existing data from JSON file or handle invalid content
            try:
                with open(json_file_path, 'r') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                data = []  # If file is empty or invalid, initialize as empty list

            # Append new form data
            data.append(form_data)

            # Write updated data back to JSON file
            with open(json_file_path, 'w') as f:
                json.dump(data, f, indent=4)

            # Redirect to the submit page
            return redirect(url_for('submit'))
        except Exception as e:
            return f"An error occurred: {str(e)}"

@app.route('/submit.html', methods=['GET'])
def submit():
    return render_template('submit.html')

# Route for prediction
@app.route('/home.html', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Collecting data from the form
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )

            # Convert data to a DataFrame
            pred_df = data.get_data_as_data_frame()
            print("DataFrame for prediction:", pred_df)

            # Predict using the pipeline
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            return render_template('home.html', results=results[0])
        except Exception as e:
            return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
