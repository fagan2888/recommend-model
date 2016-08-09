#coding=utf8



import pandas as pd
import sys
sys.path.append('shell')
import FundIndicator


if __name__ == '__main__':

    risk_portfolio_df = pd.read_csv('./tmp/risk_portfolio.csv', index_col = 'date', parse_dates = ['date'])

    risk_portfolio_dfr = risk_portfolio_df.pct_change().dropna()

    dates = risk_portfolio_df.index
    data  = []
    for i in range(1, len(dates)):
        d = dates[i]
        vdf = risk_portfolio_df.iloc[0 :i,] 
        drawdown = [] 
        for col in vdf.columns: 
            drawdown.append(FundIndicator.portfolio_drawdown(vdf[col].values)) 
        data.append(drawdown)
    drawdown_df = pd.DataFrame(data, index = dates[1:], columns = risk_portfolio_df.columns) 


    risk_portfolio_dfr.to_csv('./tmp/risk_portfolio_r.csv')
    drawdown_df.to_csv('./tmp/drawdown.csv')
