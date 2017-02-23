from flask import Flask, render_template, request
from predict_funcs import lat_long, get_datetime, get_google_distance, predicting
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
import math
import os
import pyspark as ps


# Initialize SparkSession
spark = SparkSession.builder\
    .master('local[4]')\
    .appName('model_fit')\
    .getOrCreate()

# Initialize the Flask application
app = Flask(__name__)

# Load fitted pipeline
pipe = PipelineModel.load('s3://yellowtaxidata/best_model')

# Define a route for the default URL, which loads the form


@app.route('/')
def form():
    return render_template('form_submit.html')

# Define a route for the action of the form and which type of requests this route
# is accepting: POST requests in this case


@app.route('/predict/', methods=['POST'])
def predict():
    origin = request.form['yourorigin']
    destination = request.form['yourdestination']
    flexible = request.form['flexibility']
    # Previously included minute_of_hour which was taken out due to resulting feature importance
    day_of_week, hour_of_day = get_datetime()
    prediction_0 = predicting(pipe, origin, destination, day_of_week, hour_of_day)

    # Checks if there is a quicker trip within 4 hours
    if flexible:
        predictions = []
        for x in xrange(1, 5):
            predictions.append(predicting(pipe, origin, destination, day_of_week, hour_of_day + x))
        min_prediction = min(predictions)
        if min_prediction < prediction_0:
            alert = 'Alert! You will save {} minutes if you leave in {} hour(s)'.format(prediction_0 - min_prediction, predictions.index(min_prediction) + 1)
        else:
            alert = 'Sorry, no quicker trips within 4 hours.'
        return render_template('form_action.html', destination=destination, prediction=prediction_0, alert=alert)
    else:
        return render_template('form_action.html', origin=origin, destination=destination, prediction=prediction_0, alert=None)


# Run the app
if __name__ == '__main__':
    if os.path.exists('.env'):
        print('Detected local .env, importing environment from .env...')
        for line in open('.env'):
            var = line.strip().split('=')
            if len(var) == 2:
                os.environ[var[0]] = var[1]
                print "Setting environment variable:", var[0]
                sys.stdout.flush()

    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )
