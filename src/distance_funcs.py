import math


def haversine_dist(lat_1, long_1, lat_2, long_2):
    """Calculate the haversine (crow's fly) distance between latitude/longitude pairs.

    The haversine distnace (d) is given by the formula:
        a = sin^2(delt_lat/2) + cos(lat1) * cos(lat2) * sin^2(delt_long/2)
            a is the square of half the chord length between the points
        c = 2 * atan2(sqrt(a),sqrt(1-a))
        d = R * c
        R is the Earth's radius -> approx 3,959 miles

    Note: the angles need to be in radians to pass to the trig functions

    Args: 
        lat_1, lat_2 (float): latitude of locations 1 and 2 respectively
        long_1, long_2 (float): longitude of locations 1 and 2 respectively

    Returns:
        float: haversine distance

    """

    R = 3958.754641
    lat_1, lat_2, long_1, long_2 = map(math.radians, [lat_1, lat_2, long_1, long_2])
    delt_lat = lat_2 - lat_1
    delt_long = long_2 - long_1

    a = (math.sin(.5 * delt_lat))**2 + math.cos(lat_1) * math.cos(lat_2) * (math.sin(.5 * delt_long))**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    # calculate the angular distnace in radians
    d = R * c
    return d
