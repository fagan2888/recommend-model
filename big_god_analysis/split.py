#coding=utf8


import pandas as pd


df = pd.read_csv('./intermediates/cyb.utf8.csv', index_col = 'dates' , parse_dates = 'dates')

columns = df.columns

close_columns = []
open_columns  = []
change_columns= []

for i in range(0, len(columns)):
    if i % 3 == 0:
        close_columns.append(columns[i])
    if i % 3 == 1:
        open_columns.append(columns[i])
    if i % 3 == 2:
        change_columns.append(columns[i])


close_df  = df[close_columns]
open_df   = df[open_columns]
open_df   = pd.DataFrame(open_df.values, index=close_df.index, columns=close_df.columns)
change_df = df[change_columns]
change_df   = pd.DataFrame(change_df.values, index=change_df.index, columns=close_df.columns)

#result_df = pd.DataFrame(result, index=dvalue, columns=all_columns)
#result_df.index.name = 'date'
close_df.to_csv('cyb_close.csv')
open_df.to_csv('cyb_open.csv')
change_df.to_csv('cyb_change.csv')
#print close_df



