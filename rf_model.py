from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor


def rf_model(data):

    cols_to_keep = ['pickup_longitude', 'pickup_latitude',
                    'dropoff_longitude', 'dropoff_latitude',
                    'day_of_week', 'hour_of_day', 'trip_distance',
                    'haversine_dist', 'lat_dist', 'long_dist']

    # Transform the selected features of the df into one feature column
    feature_assembler = VectorAssembler(inputCols=cols_to_keep,
                                        outputCol='features')

    # Split the data into training and test sets (10% held out for testing)
    (training_val_data, test_data) = data.randomSplit([0.9, 0.1])
    (training_data, validation_data) = training_val_data.randomSplit([0.9, 0.1])

    # training_data.persist()

    # Train a RandomForest model.
    rf = RandomForestRegressor(labelCol='log_min_duration', featuresCol='features')

    # Chain the vector assembler and the forest in a pipeline
    pipeline = Pipeline(stages=[feature_assembler, rf])

    # Train model.  This also runs the vector assembler.
    model = pipeline.fit(training_data)

    # Make predictions.
    predictions = model.transform(validation_data)

    # Evaluate model. Keep the prediction, the actual y value, and the features
    prediction_df = predictions.select('prediction', 'log_min_duration', 'features')

    # Calculate the rmse
    evaluator = RegressionEvaluator(labelCol='log_min_duration', predictionCol='prediction', metricName='rmse')

    rmse = evaluator.evaluate(prediction_df)
    print('Test Root Mean Squared Error = ' + str(rmse))

    # to return prediction in min just do math.exp(predition)

    # # Select (prediction, true label) and compute test error
    # print('Test Mean Squared Error = ' + str(valMSE))
    # print('Learned regression forest model:')
    # print(model.toDebugString())
    final_model = pipeline.fit(training_val_data)

    # Save and load model
    final_model.save('pipeline/RandomForestRegressionPipeline')

    return cols_to_keep, final_model.stages[1]


def get_importances(cols, model):
    '''
    Returns the feature importance of each feature sorted in descending order
    '''
    importances = []
    for i in xrange(len(cols)):
        importances.append((cols[i], model.featureImportances[i]))

    return sorted(importances, key=(lambda x: x[1]), reverse=True)
