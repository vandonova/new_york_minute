from distance_funcs import haversine_dist, lat_dist, long_dist
from pyspark.sql.functions import udf, avg, stddev
from pyspark.sql.types import DoubleType
from rf_model_build import rf_model, get_importances
import pyspark as ps

# credentials from the environment variables
# ACCESS_KEY = os.environ['AWS_ACCESS_KEY_ID']
# SECRET_KEY = os.environ['AWS_SECRET_ACCESS_KEY']

sc = ps.SparkContext()


def remove_outliers(df, column):
    avg_results = {}
    std_results = {}
    a = df.select(avg(column)).collect()
    s = df.select(stddev(column)).collect()
    for average, std in zip(a, s):
        avg_results.update(average.asDict())
        std_results.update(std.asDict())
    mean = avg_results['avg(' + column.strip("'") + ")"]
    stddeviation = std_results['stddev_samp(' + column.strip("'") + ")"]
    # keep only ones within 10 stddev
    clean_df = df.filter(df[column] - mean < 10 * stddeviation)
    return clean_df


def transformation_pipeline(df):
    haversine_udf = udf(lambda lat1, long1, lat2, long2: haversine_dist(lat1, long1, lat2, long2), DoubleType())

    lat_dist_udf = udf(lambda lat1, lat2: lat_dist(lat1, lat2), DoubleType())

    long_dist_udf = udf(lambda long1, long2: long_dist(long1, long2), DoubleType())

    df = df.withColumn('haversine_dist', haversine_udf(df.dropoff_latitude, df.dropoff_longitude, df.pickup_latitude, df.pickup_longitude))

    df = df.withColumn('lat_dist', lat_dist_udf(df.dropoff_latitude, df.pickup_latitude))

    df = df.withColumn('long_dist', long_dist_udf(df.dropoff_longitude, df.pickup_longitude))

    df = df.filter(df.min_duration > 0)
    df = remove_outliers(df, 'min_duration')
    return df


if __name__ == '__main__':
    # link to the S3 repository
    link = 's3a://yellowtaxidata/data/*'
    # creating an RDD...
    sqlContext = ps.SQLContext(sc)
    df = sqlContext.read.csv(link,
                             header=True,       # use headers or not
                             quote='"',         # char for quotes
                             sep=",",           # char for separation
                             inferSchema=True)  # do we infer schema or not ?

    print 'Feature engineering...'
    df = transformation_pipeline(df)
    print 'Building a rf...'
    cols, rf_model = rf_model(df)
    print 'Get feature importances...'
    importance = get_importances(cols, rf_model)
    print importance
