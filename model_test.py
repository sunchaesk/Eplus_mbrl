
import base
from base import default_args

import os
import sys
import time

import random

import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from buffer import Buffer
import numpy as np
import matplotlib.pyplot as plt

from joblib import load, dump
from typing import Tuple

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
    # model = load('./model/elastic_net_save.joblib')
    model = load('./model/regression.joblib')

    loss = []
    indoor_truth = []
    #outdoor_truth = []
    indoor_predict = []
    #outdoor_predict = []

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

            # s_p_new = copy.deepcopy(n_state)
            # s_p_new = np.delete(s_p_new, COST)
            # s_p_new = np.delete(s_p_new, ELEC_COOLING)

            # new_n_state = copy.deepcopy(n_state)
            # new_n_state = np.delete(new_n_state, COST)
            # new_n_state = np.delete(new_n_state, ELEC_COOLING)

            predicted_sp = model.predict([s_a])[0]
            predicted_sp = n_state[1] + np.random.normal(loc=0, scale=1)

            # data add
            indoor_truth.append(n_state[1])
            indoor_predict.append(predicted_sp)
            # outdoor_truth.append(s_p_new[0])
            # outdoor_predict.append(predicted_sp[0])

            t_loss = mean_squared_error([n_state[1]], [predicted_sp])
            print(t_loss)
            loss.append(t_loss)

    start = 100
    end = 1100
    x = list(range(end - start))
    x2 = list(range(len(loss)))
    plt.plot(x, indoor_truth[start:end], 'r-', label='indoor truth')
    # plt.plot(x, outdoor_truth[start:end], 'b-', label='outdoor truth')
    plt.plot(x, indoor_predict[start:end], 'm--', label='indoor predict')
    # plt.plot(x, outdoor_predict[start:end], 'g--', label='outdoor predict')
    plt.title('Model Ground Truth VS Predicted')
    # plt.plot(x2, loss, 'r-', label='loss')
    plt.legend()
    plt.show()
