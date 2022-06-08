import zmq
import time
import msgpack
import msgpack_numpy as m
import numpy as np
from operator import index
import pathlib
from posixpath import split
import numpy as np 
import pandas as pd
from pandas import DataFrame
import json
from datetime import timedelta

# import data
src_dir = pathlib.Path.cwd()  # Current directory from where the script was started.
data_dir = src_dir / 'data'

rpi = []
for csv_file in sorted(data_dir.glob('*.csv')):
     rpi.append(pd.read_csv(csv_file, header=0, parse_dates=['timestamp']))

data_fields = set(rpi[0].columns.values) - {'timestamp', 'MU_MM', 'MU_ID', 'sender_hostname'}

data_is_scaled = False
if not data_is_scaled:
    for n in range(len(rpi)):
        rpi[n]["temp-external"] = rpi[n]["temp-external"] / 10000                               # Degrees Celsius
        rpi[n]["temp-PCB"] = rpi[n]["temp-PCB"] / 10000                                         # Degrees Celsius
        rpi[n]["soil_temperature"] = rpi[n]["soil_temperature"] / 10                            # Degrees Celsius
        rpi[n]["mag_X"] = rpi[n]["mag_X"] / 1000 * 100                                          # Micro Tesla
        rpi[n]["mag_Y"] = rpi[n]["mag_Y"] / 1000 * 100                                          # Micro Tesla
        rpi[n]["mag_Z"] = rpi[n]["mag_Z"] / 1000 * 100                                          # Micro Tesla
        rpi[n]["light-external"] = rpi[n]["light-external"] / 799.4 - 0.75056                   # Lux
        rpi[n]["humidity-external"] = (rpi[n]["humidity-external"] * 3 / 4200000 - 0.1515) \
                                      / (0.006707256 - 0.0000137376 * rpi[n]["temp-external"])  # Percent
        rpi[n]["air_pressure"] = rpi[n]["air_pressure"] / 100                                   # Mili Bars
        rpi[n]["transpiration"] = rpi[n]["transpiration"] / 1000                                # Percent


def moving_average(x, w=6, mode='same'):
    if(len(x)!=0):
        return np.convolve(x, np.ones(w), 'same') / w

apply_filter = "all"
if apply_filter == "potential":
    for n in range(len(rpi)):
        rpi[n]["differential_potential_CH1"] = moving_average(rpi[n]["differential_potential_CH1"], 6)
        rpi[n]["differential_potential_CH2"] = moving_average(rpi[n]["differential_potential_CH2"], 6)
        rpi[n] = rpi[n].iloc[6:-6,:]
elif apply_filter == "all":
    for n in range(len(rpi)):
        for field in data_fields:
            rpi[n].loc[:, (field)] = moving_average(rpi[n][field], 6, 'valid')
            rpi[n] = rpi[n].iloc[10:-10,:]

# merge all dataframes into one
frames = [i for i in rpi]
rpi_concat = pd.concat(frames)
data  = json.loads(rpi_concat.to_json (orient='records'))

# Create ZMQ socket
port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind(f'tcp://*:{port}')

def morgify(topic, msg):
    return topic + ' ' + json.dumps(msg)

def send_msg(socket, data, topic='data'):
    """ Send data over ZMQ PubSub socket
    Args:
        socket: zmq.socket instance
        topic: topic to put in ZMQ topic field (str)
    """
    print(morgify(topic,data))
    return socket.send_string(morgify(topic,data))
while True:
    for messagedata in data:
        topic = "data"
        send_msg(socket, messagedata, topic)
        time.sleep(1)