def is_repair_ongoing(sensor_id):
    """
    Simulated maintenance database
    """
    ongoing_repairs = ["SENSOR_3"]
    return sensor_id in ongoing_repairs