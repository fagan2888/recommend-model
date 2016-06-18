#coding=utf8


start_date                              = '2010-01-01'


fund_measure_lookback                   = 52              #回溯52个周
fund_measure_adjust_period              = 26              #26个周重新算基金池


jensen_ratio                            = 0.5             #jensen取前50%
sortino_ratio                           = 0.5
ppw_ratio                               = 0.5
stability                               = 0.5


fixed_risk_asset_risk_lookback          = 52
fixed_risk_asset_risk_adjust_period     = 5



allocation_lookback                     = 13
allocation_adjust_period                = 13




stock_fund_measure = {}
stock_fund_label   = {}
bond_fund_measure  = {}
bond_fund_label    = {}
money_fund_measure = {}
money_fund_label   = {}
other_fund_measure = {}
other_fund_label   = {}


label_asset_df = None
stock_fund_df  = None
bond_fund_df   = None


equal_risk_asset_ratio_df = None
equal_risk_asset_df       = None


high_risk_position_df    = None
low_risk_position_df     = None
highlow_risk_position_df = None


high_risk_asset_df       = None
low_risk_asset_df        = None
highlow_risk_asset_df    = None


