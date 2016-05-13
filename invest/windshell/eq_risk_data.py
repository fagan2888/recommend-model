#coding=utf8


import string
import pandas as pd


ratio_df         = pd.read_csv('./tmp/eq_ratio.csv', index_col = 'date', parse_dates = 'date' )
allocation_df    = pd.read_csv('./tmp/5fund.csv', index_col = 'date', parse_dates = 'date' )

allocation_dfr   = allocation_df.pct_change().fillna(0)

cols = ratio_df.columns

ratio_dates     = ratio_df.index
allocation_dates = allocation_dfr.index

asset_w = {}
asset_v = {}


for code in cols:
	asset_w[code] = 0.0
for code in cols:
	asset_v.setdefault(code, [1.0])

data = {}

f = open('./tmp/eq_risk_data.csv','w')

#f.write('date,ltj,zp,lqs,wph,ydh\n')
f.write('date,ltj,zp,lqs,wph\n')

#f.write('date, largecap, smallcap, rise, oscillation, decline, growth, value, convertiblebond, SP500.SPI, SPGSGCTR.SPI, HSCI.HI\n')
#d_str = '%s, %f, %f, %f, %f ,%f, %f, %f, %f, %f, %f, %f\n'

d_str = '%s, %f, %f, %f, %f\n'


for i in range(0, len(allocation_dates)):
	d = allocation_dates[i]
	vs = []
	vs.append(d)

	for code in cols:
		r = allocation_dfr.loc[d, code] * asset_w[code]
		v = asset_v[code][-1]
		v = v * (1 + r)
		asset_v[code].append(v)
		vs.append(v)

		if d in set(ratio_dates):
			asset_w[code] = ratio_df.loc[d, code]

	f.write(d_str % tuple(vs))

f.flush()
f.close()


#print allocation_df
