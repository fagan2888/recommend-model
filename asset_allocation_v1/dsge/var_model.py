import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import  VAR, DynamicVAR
from statsmodels.tsa.base.datetools import dates_from_str
import matplotlib.pyplot as plt
plt.style.use('ggplot')

mdata = sm.datasets.macrodata.load_pandas().data
dates = mdata[['year', 'quarter']].astype(int).astype(str)
quarterly = dates['year']+'Q'+dates['quarter']
quarterly = dates_from_str(quarterly)
mdata = mdata[['realgdp', 'realcons', 'realinv']]
mdata.index = quarterly
data = np.log(mdata).diff().dropna()
model = VAR(data)
results = model.fit(2)
#results = model.fit(maxlags=15, ic = 'aic')
#print results.summary()
#results.plot_acorr()
#plt.show()
#model.select_order(15)
lag_order = results.k_ar
#print lag_order
forecast = results.forecast(data.values[-lag_order:], 5)
#print data.values[-lag_order:]
#results.plot_forecast(10)
#plt.show()
irf = results.irf(10)
#irf.plot(impulse='realgdp')
irf.plot_cum_effects(orth=False)
plt.show()
