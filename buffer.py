
import sys
import os
import random

import pickle
import numpy as np
from collections import deque, namedtuple

from typing import List, Tuple

class Buffer:
    'Fixed sized buffer for collecting model data, for Eplus building model generation'
    def __init__(self,
                 buffer_size: int,
                 data_collection_period_start: Tuple[int, int],
                 data_collection_period_end: Tuple[int, int],
                 data_collection_method: str,
                 weather_region: str):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.weather_region = weather_region
        self.data_collection_period_start = data_collection_period_start
        self.data_collection_period_end = data_collection_period_end
        # prob 'Random'
        self.data_collection_method = data_collection_method
        #self.experience = namedtuple("Experience", field_names=["state", "action", "next_state"])

    def add(self, state, action, next_state):
        e = tuple([state, action, next_state])
        self.buffer.append(e)

    def b_full(self):
        if len(self.buffer) == self.buffer_size:
            return True
        else:
            return False

    def percentage_full(self):
        return round(len(self.buffer) / self.buffer_size, 2)
