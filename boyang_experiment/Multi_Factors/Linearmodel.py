# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 18:02:57 2019

@author: yshlm
"""

#from linearmodels.datasets import wage_panel
#import pandas as pd
#data = wage_panel.load()
#year = pd.Categorical(data.year)
#data = data.set_index(['nr', 'year'])
#data['year'] = year
#print(wage_panel.DESCR)
#print(data.head())
#
#from linearmodels.panel import PooledOLS
#import statsmodels.api as sm
#exog_vars = ['black','hisp','exper','expersq','married', 'educ', 'union', 'year']
#exog = sm.add_constant(data[exog_vars])
#mod = PooledOLS(data.lwage, exog)
#pooled_res = mod.fit()
#print(pooled_res)
#
#
#
#from linearmodels.panel import RandomEffects
#mod = RandomEffects(data.lwage, exog)
#re_res = mod.fit()
#print(re_res)

# In[1]:

from linearmodels.datasets import jobtraining
data = jobtraining.load()
mi_data = data.set_index(['fcode', 'year'])

from linearmodels import PanelOLS
mod = PanelOLS(mi_data.lscrap, mi_data.hrsemp, entity_effects=True)

panel = mi_data[['lscrap','hrsemp']].to_panel().swapaxes(1,2)
lscrap = panel['lscrap']
hrsemp = panel['hrsemp']
panel

res = PanelOLS(panel[['lscrap']], panel[['hrsemp']], entity_effects=True).fit()
print(res)

res = PanelOLS(lscrap.values, hrsemp.values, entity_effects=True).fit()
print(res)

res = PanelOLS(lscrap.values, hrsemp.values, entity_effects=False).fit()
print(res)