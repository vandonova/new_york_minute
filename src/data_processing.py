from distance_funcs import haversine_dist
from pyspark.sql.functions import udf, abs
from pyspark.sql.types import DoubleType


def remove_outliers(df, column, train=True, train_mean=None, train_std=None):
    '''Remove outliers from a given column of the dataframe. 

    The function uses the training set mean and standard deviation to revome all 
    points not within 10 stddevs. It is applied to the testing set with the 
    cooresponding training set's statistics in order to prevent data leakage.

    Args:
        df (DataFrame): dataframe of either the train or test data
        column (str): column name
        train (bool); whether df is the test or train set
        train_mean (float): mean of the train set column
        train_std (float): standard deviation of the train set column 

    Returns:
        DataFrame, (and train mean and standard dev if applicable)
    '''
    if train:
        samp_mean = df.agg({column: 'mean'}).collect()[0]['avg(' + column + ')']
        samp_std = df.agg({column: 'std'}).collect()[0]['stddev(' + column + ')']
        clean_df = df.filter(abs(df[column] - samp_mean) < 10 * samp_std)
        return clean_df, samp_mean, samp_std
    else:
        clean_df = df.filter(abs(df[column] - train_mean) < 10 * train_std)
        return clean_df


def clean_data(df):
    '''
    Function that takes in the read in data and prepares it for the model.

    Cleans the data from any negative durations. Creates three columns 
    with values for haversine distance, latitude distance, and longitude distance. 

    Args: 
        df (DataFrame)
    OUTPUT: 
        df (DataFrame)

    '''
    # Filter out negatives
    df = df.filter(df.min_duration > 0)

    haversine_udf = udf(lambda lat1, long1, lat2, long2: haversine_dist(lat1, long1, lat2, long2), DoubleType())
    # Haversine with same longs is lat distance
    lat_dist_udf = udf(lambda lat1, lat2: haversine_dist(lat1, 0, lat2, 0), DoubleType())
    # Haversine with same lats is long distance
    long_dist_udf = udf(lambda long1, long2: haversine_dist(0, long1, 0, long2), DoubleType())

    # Create columns for each distance metric
    df = df.withColumn('haversine_dist', haversine_udf(df.dropoff_latitude, df.dropoff_longitude, df.pickup_latitude, df.pickup_longitude))
    df = df.withColumn('lat_dist', lat_dist_udf(df.dropoff_latitude, df.pickup_latitude))
    df = df.withColumn('long_dist', long_dist_udf(df.dropoff_longitude, df.pickup_longitude))

    return df
