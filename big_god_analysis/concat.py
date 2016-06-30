#coding=utf8
import pandas as pd

index = 'dates'
open_sh = pd.read_csv('sh_open.csv', index_col=index)
open_sz = pd.read_csv('sz_open.csv', index_col=index)
open_cyb = pd.read_csv('cyb_open.csv', index_col=index)

close_sh = pd.read_csv('sh_close.csv', index_col=index)
close_sz = pd.read_csv('sz_close.csv', index_col=index)
close_cyb = pd.read_csv('cyb_close.csv', index_col=index)

change_sh = pd.read_csv('sh_change.csv', index_col=index)
change_sz = pd.read_csv('sz_change.csv', index_col=index)
change_cyb = pd.read_csv('cyb_change.csv', index_col=index)

close_pf = close_sh.join(close_sz)
close_pf = close_pf.join(close_cyb)

open_pf = open_sh.join(open_sz)
open_pf = open_pf.join(open_cyb)

change_pf = change_sh.join(change_sz)
change_df = change_pf.join(change_cyb)
#close_pf = pd.concat([close_sh, close_sz], axis = 0)
#close_pf = pd.concat([close_pf, close_cyb], axis = 0)
#
#open_pf = pd.concat([open_sh, open_sz], axis = 0)
#open_pf = pd.concat([open_pf, open_cyb], axis = 0)
#
#change_pf = pd.concat([change_sh, change_sz], axis = 0)
#change_pf = pd.concat([change_pf, change_cyb], axis = 0)

close_pf.to_csv('close.csv', encoding='utf8')
open_pf.to_csv('open.csv', encoding='utf8')
change_pf.to_csv('change.csv', encoding='utf8')
