#coding=utf8



import pandas as pd



def risk_position():


	fund_df                   = pd.read_csv('./tmp/stock_fund.csv', index_col = 'date', parse_dates = ['date'])
	bond_fund_df              = pd.read_csv('./tmp/bond_fund.csv', index_col = 'date', parse_dates = ['date'])
	equalrisk_ratio_df        = pd.read_csv('./tmp/equalriskassetratio.csv', index_col = 'date', parse_dates = ['date'])
	#highriskposition_ratio_df = pd.read_csv('./tmp/highriskposition.csv', index_col = 'date', parse_dates = ['date'])
	highriskposition_ratio_df = pd.read_csv('./tmp/highriskposition.csv', index_col = 'date', parse_dates = ['date'])
	lowriskposition_ratio_df  = pd.read_csv('./tmp/lowriskposition.csv', index_col = 'date', parse_dates = ['date'])
	risk_portfolio_df         = pd.read_csv('./tmp/risk_portfolio.csv' , index_col  = 'date', parse_dates = ['date'])
	label_asset_df            = pd.read_csv('./tmp/labelasset.csv' , index_col  = 'date', parse_dates = ['date'])


	#print equalrisk_ratio_df
	#print highriskposition_ratio_df


	fund_df_index                = 0
	bond_fund_df_index           = 0
	equalrisk_ratio_index        = 0
	highriskposition_ratio_index = 0
	lowriskposition_ratio_index  = 0


	#risk_portfolio_index        = 0


	start_date = highriskposition_ratio_df.index[0]
	#print start_date
	equalrisk_ratio_df = equalrisk_ratio_df[equalrisk_ratio_df.index >= start_date]
	#
	dates = risk_portfolio_df.index
	start_date = dates[0]	

	label_asset_df = label_asset_df[label_asset_df.index >= start_date]
	dates = label_asset_df.index

	#print start_date 
	#print dates[0]			


	fund_codes       = {}
	bond_fund_codes  = {}
	equalrisk_ratio  = {}
	highriskposition = {}
	lowriskposition = {}

	#print fund_df
	for i in range(0, len(fund_df.index) - 1):
		if fund_df.index[i + 1] >= start_date:
			for j in range(0, len(fund_df.columns)):
				fund_codes[fund_df.columns[j]] = fund_df.iloc[i, j]
			fund_df = fund_df[fund_df.index > fund_df.index[i]]
			break
	
	#print fund_df
	for i in range(0, len(bond_fund_df.index) - 1):
		if bond_fund_df.index[i + 1] >= start_date:
			for j in range(0, len(bond_fund_df.columns)):
				bond_fund_codes[bond_fund_df.columns[j]] = bond_fund_df.iloc[i, j]
			bond_fund_df = bond_fund_df[bond_fund_df.index > bond_fund_df.index[i]]
			break


	for i in range(0, len(equalrisk_ratio_df.index) - 1):
		if equalrisk_ratio_df.index[i + 1] >= start_date:
			for j in range(0, len(equalrisk_ratio_df.columns)):
				equalrisk_ratio[equalrisk_ratio_df.columns[j]] = equalrisk_ratio_df.iloc[i, j]
			equalrisk_ratio_df = equalrisk_ratio_df[equalrisk_ratio_df.index > equalrisk_ratio_df.index[i]]
			break


	#print equalrisk_ratio
	#print fund_codes

	#print fund_df
	#print equalrisk_ratio_df.index
	#print highriskposition_ratio_df.index
	#print risk_portfolio_df.index

	#print fund_df

	all_code_position = []

	for i in range(0, len(dates)):

		d = dates[i]
		#print d

		if fund_df_index < len(fund_df.index) and d >= fund_df.index[fund_df_index]:
			for label in fund_df.columns:
				tmp_d = fund_df.index[fund_df_index]
				fund_codes[label] = fund_df.loc[tmp_d, label]
			fund_df_index = fund_df_index + 1


		if bond_fund_df_index < len(bond_fund_df.index) and d >= bond_fund_df.index[bond_fund_df_index]:
			for label in bond_fund_df.columns:
				tmp_d = bond_fund_df.index[bond_fund_df_index]
				bond_fund_codes[label] = bond_fund_df.loc[tmp_d, label]
			bond_fund_df_index = bond_fund_df_index + 1


		if equalrisk_ratio_index < len(equalrisk_ratio_df.index) and d >= equalrisk_ratio_df.index[equalrisk_ratio_index]:
			for label in equalrisk_ratio_df.columns:
				tmp_d = equalrisk_ratio_df.index[equalrisk_ratio_index]
				equalrisk_ratio[label] = equalrisk_ratio_df.loc[tmp_d, label]
			equalrisk_ratio_index = equalrisk_ratio_index + 1

		if highriskposition_ratio_index < len(highriskposition_ratio_df.index) and d >= highriskposition_ratio_df.index[highriskposition_ratio_index]:
			for label in highriskposition_ratio_df.columns:
				tmp_d = highriskposition_ratio_df.index[highriskposition_ratio_index]
				highriskposition[label] = highriskposition_ratio_df.loc[tmp_d, label]
			highriskposition_ratio_index = highriskposition_ratio_index + 1

		if lowriskposition_ratio_index < len(lowriskposition_ratio_df.index) and d >= lowriskposition_ratio_df.index[lowriskposition_ratio_index]:
			for label in lowriskposition_ratio_df.columns:
				tmp_d = lowriskposition_ratio_df.index[lowriskposition_ratio_index]
				lowriskposition[label] = lowriskposition_ratio_df.loc[tmp_d, label]
			lowriskposition_ratio_index = lowriskposition_ratio_index + 1


		#print d, fund_codes, equalrisk_ratio, highriskposition
		#print d,  bond_fund_codes, lowriskposition

		#print equalrisk_ratio
		#print d,



		for risk_rank in range(1, 11):

			high_w  = (risk_rank - 1) * 1.0 / 9
                        low_w = 1 - high_w

			#print highriskposition.keys()
			ws = {}
			for col in highriskposition.keys():


				code = None

				if col == 'GLNC':
					code = 216
				elif col == 'HSCI.HI':
					code = 000071
				elif col == 'SP500.SPI':
					code = 96001
				else:
					code = fund_codes[col]

				highriskratio   = highriskposition[col]
				risk_ratio = equalrisk_ratio[col]
				#print d, code , highriskratio * risk_ratio * high_w
				#print risk_rank / 10.0, d, code , highriskratio * risk_ratio * high_w
				#all_code_position.append((risk_rank / 10.0, d, code ,highriskratio * risk_ratio * high_w))
				#print d, code, highriskratio * risk_ratio * high_w		
				weight   = ws.setdefault(code, 0.0)
				ws[code] = weight + highriskratio * risk_ratio * high_w
	
			for col in lowriskposition.keys():

				lowriskratio = lowriskposition[col]
				code         = bond_fund_codes[col]
				#print risk_rank / 10.0, d, code, lowriskratio * low_w
				#all_code_position.append((risk_rank / 10.0, d, code, lowriskratio * low_w))
				#print col,
				
				weight   = ws.setdefault(code, 0.0)
				ws[code] = weight  + lowriskratio * low_w

			for code in ws.keys():
				w = ws[code]		
				all_code_position.append((risk_rank / 10.0, d, code, w))
		#print
	#print risk_portfolio_df
	#print fund_df

	#print all_code_position


	return all_code_position


if __name__ == '__main__':

	all_code_position = risk_position()
	for tmp in all_code_position:
		if tmp[0] == 0.8:
			print str(tmp[1]) + "\t" +  str(tmp[2]) + "\t" +  str(tmp[3])

	'''
	fund_df                   = pd.read_csv('./tmp/stock_fund.csv', index_col = 'date', parse_dates = ['date'])
	bond_fund_df              = pd.read_csv('./tmp/bond_fund.csv', index_col = 'date', parse_dates = ['date'])
	equalrisk_ratio_df        = pd.read_csv('./tmp/equalriskassetratio.csv', index_col = 'date', parse_dates = ['date'])
	#highriskposition_ratio_df = pd.read_csv('./tmp/highriskposition.csv', index_col = 'date', parse_dates = ['date'])
	highriskposition_ratio_df = pd.read_csv('./tmp/highriskposition.csv', index_col = 'date', parse_dates = ['date'])
	lowriskposition_ratio_df  = pd.read_csv('./tmp/lowriskposition.csv', index_col = 'date', parse_dates = ['date'])
	risk_portfolio_df         = pd.read_csv('./tmp/risk_portfolio.csv' , index_col  = 'date', parse_dates = ['date'])

	#print equalrisk_ratio_df
	#print highriskposition_ratio_df


	fund_df_index                = 0
	bond_fund_df_index           = 0
	equalrisk_ratio_index        = 0
	highriskposition_ratio_index = 0
	lowriskposition_ratio_index  = 0


	#risk_portfolio_index        = 0


	start_date = highriskposition_ratio_df.index[0]
	#print start_date
	equalrisk_ratio_df = equalrisk_ratio_df[equalrisk_ratio_df.index >= start_date]
	dates = risk_portfolio_df.index


	#print start_date 
	#print dates[0]			

	fund_codes       = {}
	bond_fund_codes  = {}
	equalrisk_ratio  = {}
	highriskposition = {}
	lowriskposition = {}
	
	#print fund_df

	for i in range(0, len(fund_df.index) - 1):
		if fund_df.index[i + 1] >= start_date:
			for j in range(0, len(fund_df.columns)):
				fund_codes[fund_df.columns[j]] = fund_df.iloc[i, j]
			fund_df = fund_df[fund_df.index > fund_df.index[i]]
			break
	#print fund_df

	for i in range(0, len(bond_fund_df.index) - 1):
		if bond_fund_df.index[i + 1] >= start_date:
			for j in range(0, len(bond_fund_df.columns)):
				bond_fund_codes[bond_fund_df.columns[j]] = bond_fund_df.iloc[i, j]
			bond_fund_df = bond_fund_df[bond_fund_df.index > bond_fund_df.index[i]]
			break


	for i in range(0, len(equalrisk_ratio_df.index) - 1):
		if equalrisk_ratio_df.index[i + 1] >= start_date:
			for j in range(0, len(equalrisk_ratio_df.columns)):
				equalrisk_ratio[equalrisk_ratio_df.columns[j]] = equalrisk_ratio_df.iloc[i, j]
			equalrisk_ratio_df = equalrisk_ratio_df[equalrisk_ratio_df.index > equalrisk_ratio_df.index[i]]
			break

	#print equalrisk_ratio
	#print fund_codes

	#print fund_df
	#print equalrisk_ratio_df.index
	#print highriskposition_ratio_df.index
	#print risk_portfolio_df.index

	#print fund_df

	all_code_position = []

	for i in range(0, len(dates)):

		d = dates[i]
		#print d

		if fund_df_index < len(fund_df.index) and d >= fund_df.index[fund_df_index]:
			for label in fund_df.columns:
				fund_codes[label] = fund_df.loc[d, label]
			fund_df_index = fund_df_index + 1


		if bond_fund_df_index < len(bond_fund_df.index) and d >= bond_fund_df.index[bond_fund_df_index]:
			for label in bond_fund_df.columns:
				bond_fund_codes[label] = bond_fund_df.loc[d, label]
			bond_fund_df_index = bond_fund_df_index + 1


		if equalrisk_ratio_index < len(equalrisk_ratio_df.index) and d >= equalrisk_ratio_df.index[equalrisk_ratio_index]:
			for label in equalrisk_ratio_df.columns:
				equalrisk_ratio[label] = equalrisk_ratio_df.loc[d, label]
			equalrisk_ratio_index = equalrisk_ratio_index + 1

		if highriskposition_ratio_index < len(highriskposition_ratio_df.index) and d >= highriskposition_ratio_df.index[highriskposition_ratio_index]:
			for label in highriskposition_ratio_df.columns:
				highriskposition[label] = highriskposition_ratio_df.loc[d, label]
			highriskposition_ratio_index = highriskposition_ratio_index + 1

		if lowriskposition_ratio_index < len(lowriskposition_ratio_df.index) and d >= lowriskposition_ratio_df.index[lowriskposition_ratio_index]:
			for label in lowriskposition_ratio_df.columns:
				lowriskposition[label] = lowriskposition_ratio_df.loc[d, label]
			lowriskposition_ratio_index = lowriskposition_ratio_index + 1


		#print d, fund_codes, equalrisk_ratio, highriskposition
		#print d,  bond_fund_codes, lowriskposition

		#print equalrisk_ratio
		#print d,



		for risk_rank in range(1, 11):

			high_w  = (j - 1) * 1.0 / 9
                        low_w = 1 - high_w

			for col in highriskposition.keys():

				#print col,

				code = None

				if col == 'GLNC':
					code = 159937
				elif col == 'HSCI.HI':
					code = 513600
				elif col == 'SP500.SPI':
					code = 513500
				else:
					code = fund_codes[col]

				highriskratio   = highriskposition[col]
				risk_ratio = equalrisk_ratio[col]
				#print d, code , highriskratio * risk_ratio * high_w
				#print risk_rank / 10.0, d, code , highriskratio * risk_ratio * high_w
				all_code_position.append((risk_rank / 10.0, d, code ,highriskratio * risk_ratio * high_w))	
					
			for col in lowriskposition.keys():

				lowriskratio = lowriskposition[col]
				#print risk_rank / 10.0, d, code, lowriskratio * low_w
				all_code_position.append((risk_rank / 10.0, d, code, lowriskratio * low_w))
				#print col,


		#print
	#print risk_portfolio_df
	#print fund_df

	print all_code_position

	'''
