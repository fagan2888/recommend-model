benchmark <- read.csv('./data/bond_benchmark.csv', header = T)
data <- read.csv("./data/fund_values.csv",header = T)
len = length(data)

result <- c('','中证国债', '中证企业债', '中证转债')
i = 2
while(i <= len){
  c1 <- cor(na.omit(cbind(data[i], benchmark$中证国债)))
  c2 <- cor(na.omit(cbind(data[i], benchmark$中证企业债)))
  c3 <- cor(na.omit(cbind(data[i], benchmark$中证转债)))
  res <- c(colnames(data[i]), c1[1,2], c2[1,2], c3[1,2])
  result <- rbind(result, res)
  i <- i + 1
}

write.csv(na.omit(result), './data/cor.csv')