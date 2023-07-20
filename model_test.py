
import base
from base import default_args

import os
import sys
import time

import sklearn
from sklearn.metrics import mean_squared_error
from buffer import Buffer
import numpy as np
import matplotlib.pyplot as plt

from joblib import load, dump

import copy

def _rescale(
    n: int,
    range1: Tuple[float, float],
    range2: Tuple[float, float]
) -> float:
    action_nparray = np.linspace(range2[0], range2[1], (range1[1] - range1[0]))
    #print(action_nparray)
    return action_nparray[n]

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

if __name__ == "__main__":
    print(default_args)
    env = base.EnergyPlusEnv(default_args)

    # load trained model
    model = load('elastic_net_save.joblib')

    loss = []

    for episode in range(1):
        state = env.reset()
        done = False
        steps = 1

        while not done:
            action = env.action_space.sample()
            ret = n_state, reward, done, truncated, info = env.step(action)

            action_scaled = _rescale(
                n = int(action),
                range1=(0, env.action_space.n),
                range2=(20, 26)
            )

            s_a = np.append(state, action_scaled)
            s_a = np.delete(s_a, COST)
            s_a = np.delete(s_a, ELEC_COOLING)

            s_p_new = copy.deepcopy(n_state)
            s_p_new = np.delete(s_p_new, COST)
            s_p_new = np.delete(s_p_new, ELEC_COOLING)

            predicted_sp = model.predict(s_a)

            loss.append(mean_squared_error(s_p_new, n_state))

    x = list(range(len(loss)))
    plt.plot(x, loss, 'r-', label='loss')
    plt.title('Loss for {} model predictor'.format('Elastic Net'))
    plt.show()
