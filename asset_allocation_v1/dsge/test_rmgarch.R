library(quantmod)
library(rugarch)
library(rmgarch)
# Pull the data
k <- 10 # how many years back?
end<- format(Sys.Date(),"%Y-%m-%d") 
end_training <- "2015-01-01"
start<-format(Sys.Date() - (k*365),"%Y-%m-%d")
sym = c('SPY', 'TLT', 'IEF')
l <- length(sym)
dat0 = getSymbols(sym[1], src="yahoo", from=start, to=end, auto.assign = F, warnings = FALSE, symbol.lookup = F)
n = NROW(dat0)  
dat = array(dim = c(n,NCOL(dat0),l)) ; ret = matrix(nrow = n, ncol = l) 
for (i in 1:l){
    dat0 = getSymbols(sym[i], src="yahoo", from=start, to=end, auto.assign = F, warnings = FALSE, symbol.lookup = F)
    dat[1:n,,i] = dat0 
    ret[2:n,i] = (dat[2:n,4,i]/dat[1:(n-1),4,i] - 1) # returns, close to close, percent
}
ret <- 100*na.omit(ret)# Remove the first observation
time <- index(dat0)[-1]
colnames(ret) <- sym
nassets <- dim(ret)[2]
TT <- dim(ret)[1]
# Fit a ghreshold garch process
gjrtspec <- ugarchspec(mean.model=list(armaOrder=c(0,0)), variance.model =list(model = "gjrGARCH"),distribution="std") 
# dcc specification - GARCH(1,1) for conditional correlations
dcc_spec = dccspec(uspec = multispec(replicate(l, gjrtspec)), distribution = "mvt")
# Fit DCC
garchdccfit = dccfit(dcc_spec, ret, fit.control=list(scale=TRUE)) 
# print(garchdccfit) # If you want to see the information - parameters etc.
slotNames(garchdccfit) # To see what is in the object
names(garchdccfit@model) # To see what is in the object
names(garchdccfit@mfit) # To see what is in the object
# plot(garchdccfit) # All kind of cool plots
