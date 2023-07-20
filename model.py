
import sys
import os

from buffer import Buffer
import pickle
import numpy as np
from joblib import dump,load

from sklearn.linear_model import ElasticNet

def append_scalars_to_array(arr_list, scalar_list):
    return [np.concatenate((arr, [scalar])) for arr, scalar in zip(arr_list, scalar_list)]

def delete_index_to_array(arr_list, index):
    mask = np.ones(arr_list[0].shape, dtype=bool)
    mask[index] = False
    result = [arr[mask] for arr in arr_list]
    return result

OUTDOOR_TEMP = 0
INDOOR_TEMP = 1
DIFFUSE_SOLAR_LDF = 2
DIFFUSE_SOLAR_SDR = 3
SITE_DIRECT_SOLAR = 4
SITE_HORZ_INFRARED = 5
ELEC_COOLING = 6
HOUR = 7
DAY_OF_WEEK = 8
DAY = 9
MONTH = 10
COST_RATE = 11
COST = 12

regr = ElasticNet(alpha=0.1)

pf = open('training_data.pt', 'rb')
buf = pickle.load(pf)
pf.close()

X_train = []
y_train = []

# Convert buf.buffer into a NumPy array
buf_array = np.array(buf.buffer,dtype=object)
# Extract the columns using slicing
s = buf_array[:, 0]
a = buf_array[:, 1]
s_p = buf_array[:, 2]

s_a = append_scalars_to_array(s, a)
# have to delete from outermost index
s_a = delete_index_to_array(s, COST)
s_a = delete_index_to_array(s, ELEC_COOLING)
s_p = delete_index_to_array(s_p, COST)
s_p = delete_index_to_array(s_p, ELEC_COOLING)

if __name__ == "__main__":
    print('eg:',buf.buffer[0], len(buf.buffer[0][0]))
    print('s:', s[1], len(s[1]))
    print('s_p:', s_p[1], len(s_p[1]))
    # start training ElasticNet regressor
    print('######### TRAINING STARTED #######')
    regr.fit(s_a, s_p)
    print('######### TRAINING COMPLETED #####')
    print('SAVING MODEl...')
    dump(regr, 'elastic_net_save.joblib')
