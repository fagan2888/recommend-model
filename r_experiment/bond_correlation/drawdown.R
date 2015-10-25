library(tseries)

data <- read.csv("./data/fund_values.csv")
data <- data[,2:length(data)]
result <- c("code",'r','maxdrawdown','kama')
codes   <- colnames(data)

for (code in codes){
  values <- data[,code]
  values <- na.omit(values)
  
  if(!is.null(values)){
    len <- length(values)
    r <- (values[len] / values[1]) ^ (1 / (len / (252))) - 1
    #print(r)
    mdd <- maxdrawdown(values)
    md  <- mdd$maxdrawdown / values[mdd$from[length(mdd$from)]]
    print(md)
    #print(mdd)
    #print(values)
    result <- rbind(result, c(code, r, md, r / md))
    #print(code)
    #print(r)
    #print(md)
    #print(r / md)
    
  }
}
write.csv(na.omit(result), './data/mdd.csv')
#print(na.omit(data$X81))
#print(maxdrawdown(na.omit(data$X81)))
#mdd <- maxdrawdown(na.omit(dax))