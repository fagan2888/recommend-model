#coding=utf8


import MySQLdb


conn = MySQLdb.connect(host='dev.mofanglicai.com.cn', port=3306, user='jiaoyang', passwd='q36wx5Td3Nv3Br2OPpH7', db='recommend', charset='utf8')

cur = conn.cursor()
cur.execute('delete from fund_portfolios')
cur.execute('delete from fund_portfolio_expect_trends')
cur.execute('delete from fund_portfolio_histories')
cur.execute('delete from fund_portfolio_industry_dispersions')
cur.execute('delete from fund_portfolio_liquidities')
cur.execute('delete from fund_portfolio_risks')
cur.execute('delete from fund_portfolio_risk_vs_returns')
cur.execute('delete from fund_portfolio_weights')

cur.close()
conn.commit()
conn.close()
