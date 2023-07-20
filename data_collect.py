
import sys
import os

import numpy as np
import pickle
from typing import Tuple

import base
from buffer import Buffer

import time

default_args = {'idf': './in.idf',
                'epw': './weather.epw',
                'csv': True,
                'output': './output',
                'timesteps': 1000000.0,
                'num_workers': 2,
                'annual': False,# for some reasons if not annual, funky results
                }

def _rescale(
        n: int,
        range1: Tuple[float, float],
        range2: Tuple[float, float]
) -> float:
    action_nparray = np.linspace(range2[0], range2[1], (range1[1] - range1[0]))
    return action_nparray[n]

def _add_10_minutes(inp):
    year, month, day, hour, minute = inp

    # Calculate the total number of minutes
    total_minutes = (hour * 60) + minute + 10

    # Calculate the new hour and minute values
    new_hour = total_minutes // 60
    new_minute = total_minutes % 60

    # Handle hour and day overflow
    if new_hour >= 24:
        new_hour %= 24
        day += 1

    # Handle month and year overflow
    if month in [1, 3, 5, 7, 8, 10, 12] and day > 31:
        day = 1
        month += 1
    elif month in [4, 6, 9, 11] and day > 30:
        day = 1
        month += 1
    elif month == 2:
        if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
            if day > 29:
                day = 1
                month += 1
        else:
            if day > 28:
                day = 1
                month += 1

    # Handle minute overflow and represent 0 minutes as 60
    if new_minute == 0:
        new_minute = 60
        new_hour -= 1

    if new_hour == -1:
        new_hour = 23

    return (year, month, day, new_hour, new_minute)

def _get_cost_signal(day_of_week, hour, minute):
    '''get cost signal at given time. @param: minute is not used'''
    if day_of_week in [1, 7]:
        # weekend pricing
        if hour in range(0, 7) or hour in range(23, 24 + 1): # plus one is to include 7
            #self.next_obs['cost_rate'] = 2.4
            return 2.4
        elif hour in range(7, 23):
            #self.next_obs['cost_rate'] = 7.4
            return 7.4
    else:
        if hour in range(0, 7) or hour in range(23, 24 + 1):
            #self.next_obs['cost_rate'] = 2.4
            return 2.4
        elif hour in range(7, 16) or hour in range(21, 23):
            #self.next_obs['cost_rate'] = 10.2
            return 10.2
        elif hour in range(16, 21):
            #self.next_obs['cost_rate'] = 24.0
            return 24.0

if __name__ == "__main__":
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
    COST_RATE_NEXT_TIMESTEP = 13
    #YEAR = 13


    buf = Buffer(5000000000,
                 tuple([5, 1]),
                 tuple([8, 31]),
                 'random',
                 'Rochester International Arpt,MN,USA')

    load = True
    if load:
        p_f = open('training_data.pt', 'rb')
        buf = pickle.load(p_f)
        p_f.close()
        print('#######################')
        print('LOADING FROM training_data.pt...')
        print('BUF SIZE:', len(buf.buffer))
        print('#######################')
        time.sleep(2)

    start = time.time()
    env = base.EnergyPlusEnv(default_args)
    iterations = 0
    steps = 0
    while not buf.b_full() and iterations <= 160 - 100:
        state = env.reset()
        done = False
        print('------------------COMPLETION: {}% / Iteration {}-----------------'.format(buf.percentage_full() * 100, iterations))
        while not done:
            steps += 1
            # choose a random action
            action = env.action_space.sample()
            action_scaled = _rescale(
                n = int(action),
                range1=(0, env.action_space.n),
                range2=(15,30)
            )
            ret = n_state, reward, done, truncated, info = env.step(action)
            #print('s', state[1], 'a', action_scaled, 'sp', n_state[1])

            # NOTE: preprocess (s, a, s')

            buf.add(state, action_scaled, n_state)
            #print('ACTION', action_scaled)
            state = n_state


        #
        iterations += 1
        if iterations % 5 == 0 and iterations != 0:
            pickle_file = open('training_data.pt', 'wb')
            pickle.dump(buf, pickle_file)
            pickle_file.close()

    end = time.time()
    print('COMPLETED')
    print('DATA COLLECTION TOOK {} SECONDS'.format(start - end))
