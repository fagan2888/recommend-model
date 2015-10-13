benchmark <- read.csv('./data/beta_benchmark.csv', header = T)
data <- read.csv("./data/beta_fund_value.csv",header = T)
len = length(data)
cols = colnames(data)

result <- c('','中证国债', '中证企业债', '中证转债')
alpha_results = c('','')
i = 2
while(i <= len){
  
  d <- t(data[cols[i]])
  c1 <- cov(na.omit(cbind(data[cols[i]], benchmark$中证国债)))
  c2 <- cov(na.omit(cbind(data[cols[i]], benchmark$中证企业债)))
  c3 <- cov(na.omit(cbind(data[cols[i]], benchmark$中证转债)))
  beta1 <- c1[1,2] / c1[2,2]
  beta2 <- c2[1,2] / c2[2,2]
  beta3 <- c3[1,2] / c3[2,2]
  alpha <- mean(d) - beta1 * mean(benchmark$中证国债) - beta2 * mean(benchmark$中证企业债) - beta3 * mean(benchmark$中证转债)
  
  res <- c(colnames(data[i]), c1[1,2] / c1[2,2], c2[1,2] / c2[2,2], c3[1,2] / c3[2,2])
  
  result <- rbind(result, res)
  alpha_results <- rbind(alpha_results, c(cols[i], alpha))
  i <- i + 1
}

write.csv(na.omit(result), './data/beta.csv')
write.csv(na.omit(alpha_results), './data/alpha.csv')