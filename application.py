from flask import Flask, request, render_template, redirect, url_for, session
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application
app.secret_key = "supersecret"  # Needed for session

@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Collect form inputs
        gender = request.form.get('gender')
        ethnicity = request.form.get('race/ethnicity')
        parental_level = request.form.get('parental_level_of_education')
        lunch = request.form.get('lunch')
        test_prep = request.form.get('test_preparation_course')
        reading_score = float(request.form.get('reading score'))
        writing_score = float(request.form.get('writing score'))

        # Create custom data
        data = CustomData(
            gender=gender,
            ethnicity=ethnicity,
            parental_level_of_education=parental_level,
            lunch=lunch,
            test_preparation_course=test_prep,
            reading_score=reading_score,
            writing_score=writing_score
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        # Prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Store in session temporarily
        session['results'] = results[0]
        session['gender'] = gender
        session['ethnicity'] = ethnicity
        session['parental_level'] = parental_level
        session['lunch'] = lunch
        session['test_prep'] = test_prep
        session['reading_score'] = reading_score
        session['writing_score'] = writing_score

        # Redirect so refresh won’t resubmit POST
        return redirect(url_for('predict_datapoint'))

    # GET request
    results = session.pop('results', None)  # Clear after showing once
    return render_template(
        'index.html',
        results=results,
        gender=session.pop('gender', None),
        ethnicity=session.pop('ethnicity', None),
        parental_level=session.pop('parental_level', None),
        lunch=session.pop('lunch', None),
        test_prep=session.pop('test_prep', None),
        reading_score=session.pop('reading_score', None),
        writing_score=session.pop('writing_score', None)
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
