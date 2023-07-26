
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# Assuming you have the data in the following format:
# X: a 2D array containing input features [outdoor_temp, horizontal_infrared, time]
# y: a 1D array containing the corresponding indoor temperature

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
v = buf_array[:, 2]
y = np.asfarray(v)
print('y:', y.dtype)
#sys.exit(1)

X = append_scalars_to_array(s, a)
# have to delete from outermost index
X = delete_index_to_array(s, COST)
X = delete_index_to_array(s, ELEC_COOLING)
X = np.array(X)

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
print(X.shape[1])
