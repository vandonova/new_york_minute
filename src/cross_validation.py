from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from data_processing import clean_data, remove_outliers
# from pyspark import SparkContext, SparkConf - no longer needed in 2.0
from pyspark.sql import SparkSession


def split_data(df):
    """Cleans, splits and removes outliers from test and training set.

    Args:
        df (DataFrame)

    Returns:
        train (DataFrame)
        test (DataFrame)

    """
    clean_df = clean_data(df)
    train, test = clean_df.randomSplit([0.9, 0.1], seed=123)
    train, dur_mean, dur_std = remove_outliers(train, 'min_duration')
    test = remove_outliers(test, 'min_duration', False, dur_mean, dur_std)
    train.persist()
    test.persist()
    return train, test


def create_rf_pipeline():
    """Wrapper function that creates a pipeline including a Vector Assembler
    and a Random Forest Regressor.

    Args:
        None

    Returns:
        cols_to_keep (list): a list of the names of feature the model will train on
        pipeline: a pipeline of the feature assembler and the random forest
        param_grid: a grid of parameters for the grid search step

    """
    cols_to_keep = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'day_of_week', 'hour_of_day', 'trip_distance', 'haversine_dist', 'lat_dist', 'long_dist']

    feature_assembler = VectorAssembler(inputCols=cols_to_keep, outputCol='features')

    rf = RandomForestRegressor(labelCol='log_min_duration', featuresCol='features')

    pipeline = Pipeline(stages=[feature_assembler, rf])

    param_grid = ParamGridBuilder()\
        .addGrid(rf.numTrees, [20, 50, 100]) \
        .addGrid(rf.maxDepth, [5, 10, 15])\
        .build()

    return cols_to_keep, pipeline, param_grid


def cross_validate(train, estimator, param_grid, evaluator, train_ratio=.8):
    """Function that uses TrainValidationSplit to cross validate and tune 
    hyper-parameters. It then returnd a fitted model with the best 
    combination of parameters. 

    Args:
        train (DataFrame): the training data
        estimator: in this case a Pipeline object
        evaluator: how the model will be evaluated 
        train_ratio (float): the fraction of the data used to train the model

    Returns:
        pipeline: a TrainValidationSplitModel object fitted with the best parameters

    """
    print 'setting tvs'
    tvs = TrainValidationSplit(estimator=estimator,
                               estimatorParamMaps=param_grid,
                               evaluator=evaluator,
                               trainRatio=train_ratio)
    print 'fitting pipeline'
    pipeline = tvs.fit(train)
    print 'saving pipeline'
    pipeline.bestModel.save('s3://yellowtaxidata/best_model')

    return pipeline


def evaluate_model(test, estimator, evaluator):
    """Predicts and evaluates the model on the test set and returns the rmse.

    Note: rmse returns on average how many log(minutes) the model is off by,
    in order to converst into minutes we must take np.exp(rmse)

    Args:
        train (DataFrame): the training data
        estimator: in this case a Pipeline object
        evaluator: how the model will be evaluated 
        train_ratio (float): the fraction of the data used to train the model

    Returns:
        pipeline: a TrainValidationSplitModel object fitted with the best parameters

    """
    predictions = estimator.transform(test)
    prediction_df = predictions.select('prediction', 'log_min_duration', 'features')
    rmse = evaluator.evaluate(prediction_df)

    return rmse


def get_importances(cols, model):
    """Returns the feature importance of each feature sorted in descending order.

    Args:
        cols_to_keep (list): a list of the feature names 
        model: the Random Forest Regressor model

    Returns:
        pipeline (list): a list of tuples of feature name and importance

    """
    importances = []
    for i in xrange(len(cols)):
        importances.append((cols[i], model.featureImportances[i]))

    return sorted(importances, key=(lambda x: x[1]), reverse=True)

if __name__ == '__main__':
    spark = SparkSession.builder\
        .master('local[4]')\
        .appName('model_fit')\
        .getOrCreate()

    link = 's3a://yellowtaxidata/data/*'

    # Reading in data from S3 bucket
    df = spark.read.csv(link, header=True, quote='"', sep=",", inferSchema=True)
    train, test = split_data(df)

    cols, pipeline, param_grid = create_rf_pipeline()

    evaluator = RegressionEvaluator(labelCol='log_min_duration', predictionCol='prediction', metricName='rmse')

    pipeline = cross_validate(train, pipeline, param_grid, evaluator)

    rmse = evaluate_model(test, pipeline, evaluator)

    feature_importance = get_importances(cols, pipeline.bestModel.stages[1])
