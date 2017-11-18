import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import  VAR, DynamicVAR
from statsmodels.tsa.base.datetools import dates_from_str
import matplotlib.pyplot as plt
import pandas.util.testing as ptest
import datetime

np.random.seed(1)
ptest.N = 500
data = ptest.makeTimeDataFrame().cumsum(0)

var = DynamicVAR(data, lag_order=2, window_type='expanding')
var.plot_forecast(2)
plt.show()
