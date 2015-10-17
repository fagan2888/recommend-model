source('./shell/gmmhmm.R')
source('./shell/mysql.R')


fund_codes <- c(000051, 160119)
#fund_codes <- c(000051)
start_date <- '2010-01-01'
end_date <- Sys.Date()
print(end_date);

res <- mysql_fund_values(fund_codes, start_date, end_date)
#print(res)
#res <- res[1:nrow(res) -1]

res_weekly <- res[endpoints(res, on = "weeks")]
#res_sma2 <- EMA(res_weekly, 2)
#res_weekly <- na.omit(cbind.xts(res_weekly, res_sma2))
hmm <- regime_gmmhmm(res_weekly, 1, 3)


#print(hmm)
#write.csv(as.data.frame(hmm$hmm_yhat[1:nrow(hmm$hmm_yhat) - 1]), './data/hs_status')
write.csv(as.data.frame(hmm$hmm_yhat), './data/hs_status')
write.csv(as.data.frame(hmm$sharpe_ratio), './data/sharpe')
write.csv(as.data.frame(hmm$hmm$model$transition), './data/transition')
