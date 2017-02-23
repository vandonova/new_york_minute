from distance_funcs import haversine_dist
import datetime
import googlemaps
import requests
import os


def lat_long(address):
    """Makes a request to the Google api for the exact latitude and longitude of the user's requested location.

    Example: lat_long('Times Square')

    Args:
        address (str): user's typed input, either an address or location name

    Returns:
        lat, long (float): latitude and longitude of the requested location

    """
    gmaps_geocoding = os.getenv('GMAPS_GEOCODING_API_KEY')
    address_result = requests.get('https://maps.googleapis.com/maps/api/geocode/json?address={}&key={}'.format(address.replace(' ', '+'), gmaps_geocoding)).json()

    lat = address_result['results'][0]['geometry']['location']['lat']
    lng = address_result['results'][0]['geometry']['location']['lng']
    return lat, lng


def get_datetime():
    """Gets the current day of the week and hour.

    Args:
        None

    Returns:
        day_of_week, hour_of_day (int): day of the week and the hour

    """
    t = datetime.datetime.now()
    day_of_week = t.weekday()
    hour_of_day = t.hour
    return day_of_week, hour_of_day


def get_google_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude):
    """Returns the driving distance between two lat/long pairs.

    Args:
        pickup longitude, pickup latitude (float)
        dropoff longitude, dropoff latitude (float)

    Returns:
        car__dist (float)

    """
    gmaps_distance_key = os.getenv('GMAPS_DISTANCE_API_KEY')
    gmaps = googlemaps.Client(key=gmaps_distance_key)
    car_dist_str = gmaps.distance_matrix((pickup_latitude, pickup_longitude), (dropoff_latitude, dropoff_longitude), mode='driving', units='imperial')['rows'][0]['elements'][0]['distance']['text']
    car_dist = float(car_dist_str.strip(' mi'))
    return car_dist


def predicting(predictor, origin, destination, day_of_week, hour_of_day):
    """Getting the model prediction rounded up to the nearest minute.

    Args:
        predictor: a fitted model
        origin (str): user's typed input, either an address or location name
        destination (str): user's typed input, either an address or location name
        day_of_week (int): day of the week for TODAY    
        hour_of_day (int): current hour

    Returns:
        prediction (int): predicted trip duration

    """
    origin_lat, origin_long = lat_long(origin)
    dest_lat, dest_long = lat_long(destination)
    google_dist = get_google_distance(origin_lat, origin_long, dest_lat, dest_long)
    hav_dist = haversine_dist(origin_lat, origin_long, dest_lat, dest_long)
    d_lat = lat_dist(origin_lat, 0, dest_lat, 0)
    d_long = long_dist(0, origin_long, 0, dest_long)

    col_names = ['pickup_longitude', 'pickup_latitude',
                 'dropoff_longitude', 'dropoff_latitude',
                 'day_of_week', 'hour_of_day', 'trip_distance',
                 'haversine_dist', 'lat_dist', 'long_dist']

    data_point = sc.parallelize([[origin_long, origin_lat, dest_long, dest_lat, day_of_week, hour_of_day, minute_of_hour, google_dist, hav_dist, d_lat, d_long]]).toDF(col_names)

    prediction = int(math.ceil(math.exp(pipe.transform(data_point).select('prediction').toPandas()['prediction'][0])))

    return prediction
