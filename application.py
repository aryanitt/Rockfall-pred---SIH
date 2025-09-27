from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Collect form data
        data = CustomData(
            slope_angle=float(request.form.get('slope_angle', 0)),
            rock_density=float(request.form.get('rock_density', 0)),
            vibration=float(request.form.get('vibration', 0)),
            rainfall=float(request.form.get('rainfall', 0)),
            temperature=float(request.form.get('temperature', 0)),
            soil_moisture=float(request.form.get('soil_moisture', 0)),
            joint_spacing=float(request.form.get('joint_spacing', 0)),
            weathering=request.form.get('weathering')
        )

        # Convert into DataFrame
        pred_df = data.get_data_as_dataframe()

        # Prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Convert prediction to human-readable output
        prediction_label = "⚠️ Rockfall Risk" if results[0] == 1 else "✅ No Risk"

        return render_template('home.html', results=prediction_label)


if __name__ == "__main__":
    app.run(debug=True)
