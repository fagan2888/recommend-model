data <- read.csv("./data/beta_fund_value.csv",header = T)
len = length(data)
cols = colnames(data)

result <- c('code','shape','r','sd')

i <- 2
while(i <= len){
  
  d   <- t(data[cols[i]])
  var <- sd(d)
  r   <- mean(d)
  res <- c(cols[i], r / var ,r ,var)
  
  result <- rbind(result, res)
  
  i <- i + 1
}

write.csv(na.omit(result), './data/shape.csv')