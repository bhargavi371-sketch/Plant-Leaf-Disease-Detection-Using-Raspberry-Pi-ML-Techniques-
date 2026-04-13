import random

def read_dht11_sensor():
    # Simulate values
    temperature = random.uniform(25, 35)
    humidity = random.uniform(40, 70)
    return round(temperature, 2), round(humidity, 2)

def read_soil_moisture_sensor():
    return round(random.uniform(30, 70), 2)