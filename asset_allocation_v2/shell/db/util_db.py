#coding=utf-8
'''
Created on: May. 14, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def to_list(x):

    if isinstance(x, str):
        x = [x]
    elif isinstance(x, (tuple, set)):
        x = list(x)
    elif isinstance(x, dict):
        x = list(x.values())
    else:
        if isinstance(x, (pd.Index, pd.Series, pd.DataFrame)):
            x = x.values
        if isinstance(x, np.ndarray):
            x = x.reshape(-1).tolist()

    return x

