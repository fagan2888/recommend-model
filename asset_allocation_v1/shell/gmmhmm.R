require(mhsmm)
require(mclust)
require(xts)
require(PortfolioAnalytics)
require(PerformanceAnalytics)
require(TTR)


regime_gmmhmm <- function(price_data, target_index = 1, nstate = 0) {
  prices <- na.omit(price_data);
  ret <- ROC(prices, n = 1, type = "continuous")
  ret1 <- ROC(prices, n = 5, type = "continuous")
  ret2 <- ROC(prices, n = 20, type = "continuous")
  ret3 <- ROC(prices, n = 60, type = "continuous")
  ret4 <- ROC(prices, n = 200, type = "continuous")
  #ret <- Return.calculate(prices[endpoints(prices, on=period)], method = "log")
  ret <- na.omit(cbind(ret, ret1, ret2, ret3, ret4))
  #ret <- na.omit(cbind.xts(ret, ret1, ret2, ret3, ret4))
  gmm <- gmm_training(ret, nstate) # GMM 捕捉指定数量的市场状态， 并生成相应的参数；
  #print(ret[,target_index]);
  hmm <- hmm_training(gmm, data_training = ret, ret_target = ret[, target_index])
  return(hmm)
}


# GMM Training Model 
# using return data as input data, calibrate n-centroids gaussian mixture model
# and find the most-likely model with the lowerest BIC setup
# input -
#     data_training: the training data set (as Data.Frame )
#     nstate: default = 0. the number of states to specify. If nstate=0, the function will find the 
#             optimal number of nstates. 
# output - gmm model, including 
#     J: (number of centroids)
#     b: (mean, vcv)
#     gmm: (gmm model calibrated)
##########################################################################
gmm_training <- function(data_training, nstate=0) {
  output <- list();
  
  if (nstate == 0) {
    mm_model <- Mclust(data_training);
  }
  else {
    mm_model <- Mclust(data_training, G = nstate)
  }
  mm_output <- summary(mm_model);
  print(mm_output);
  
  ### determine whether training dataset is a single serie or multiples
  n_serie <- ncol(data_training)  
  
  #### creating the HMM model
  J <- mm_output$G
  #print(paste("Nr of Regimes ", J))
  initial <- rep(1/J, J)
  P <- matrix(rep(1/J, J*J), nrow=J)
  
  if (n_serie == 1) {
    mean <- as.numeric(mm_output$mean)
    vcv <- as.numeric(mm_output$variance)
  }
  else
  {
    mean <- list()
    vcv <- list()
    for (j in 1:J){
      mean[[j]] <- mm_output$mean[, j]
      vcv[[j]] <- mm_output$variance[,,j]
    }
  }

  b <- list()
  b$mu <- mean
  b$sigma <- vcv

  output$gmm <- mm_model
  output$b <- b;
  output$J <- J;
  output$initial <- initial;
  output$P <- P;
  output$mean <- mean;
  output$vcv <- vcv;
  return(output);
}


hmm_training <- function(gmm, data_training, data_testing = NULL, ret_target) {
  output <- list();
  
  ### determine whether training dataset is a single serie or multiples
  n_serie <- ncol(data_training)  
  
  #### training HMM model
  if (n_serie == 1) {
    hmm_model <- hmmspec(init=gmm$initial, trans=gmm$P, parms.emission = gmm$b, dens.emission = dnorm.hsmm)
    hmm_fitted <- hmmfit(as.numeric(data_training), hmm_model, mstep = mstep.norm)
    #train <- simulate(hmm_model, nsim = 500, seed = 1234, rand.emis = rnorm.hsmm)
    #hmm_fitted <- hmmfit(train, hmm_model, mstep = mstep.norm)
  }
  else {
    hmm_model <- hmmspec(init=gmm$initial, trans=gmm$P, parms.emission = gmm$b, dens.emission = dmvnorm.hsmm)
    hmm_fitted <- hmmfit(data_training, hmm_model, mstep = mstep.mvnorm)
    #train <- simulate(hmm_model, nsim = 500, seed = 1234, rand.emis = rmvnorm.hsmm)
    #hmm_fitted <- hmmfit(train, hmm_model, mstep = mstep.mvnorm)
  }

  #print(gmm$mean);
  yhat <- predict(hmm_fitted,data_training)
  print(tail(yhat, 10))
  #print(hmm_fitted$model);
  data_training$state <- yhat$s;
  stats = data_training$state
  output$stats <- stats;
  #print(tail(stats, 100))
  #print(hmm_fitted$yhat);
  #print("hmm fitting")
  #### Predict future regime
  regime <- tail(hmm_fitted$yhat, 1);
  output$hmm <- hmm_fitted

  
  ############################################################
  #### In the training set, the regimes and returns
  #yhat_train <- stats
  #ret_training_regime <- list()
  #for (k in 1:gmm$J) {
  #  ret_training_regime[[k]] <- ret_target * (yhat_train == k)
  #}
  #ret_training_regime <- do.call(cbind, ret_training_regime)
  
  #output$hmm_yhat <- yhat_train
  #output$hmm_ret_regime <- ret_training_regime
  #output$hmm_predict_regime <- tail(output$hmm_yhat, 1);
  
  #print(sum(ret_training_regime))

  ### calculate the risk measures 
  #sharpe_training_regime_vol <- SharpeRatio.annualized(ret_training_regime)[1,]
  #max_sharpe_regime <- match(max(sharpe_training_regime_vol), sharpe_training_regime_vol)
  #calmar_training_regime <- CalmarRatio(ret_training_regime)
  #max_calmar_regime <- match(max(calmar_training_regime), calmar_training_regime)
  #sortino_training_regime <- SortinoRatio(ret_training_regime)
  #max_sortino_regime <- match(max(sortino_training_regime), sortino_training_regime)
  #output$hmm_ret_regime_annualized <- Return.annualized(ret_training_regime)

  #print(sharpe_training_regime_vol)


  #output$sharpe_ratio <- sharpe_training_regime_vol;
  #output$sharpe_ratio_max_regime <- max_sharpe_regime;
  #output$calmar_ratio <- calmar_training_regime;
  #output$calmar_ratio_max_regime <- max_calmar_regime;
  #output$sortino_ratio <- sortino_training_regime;
  #output$sortino_ratio_max_regime <- max_sortino_regime;

  return(output);

}

ret <- read.csv('tmp/nav.csv', header = TRUE, sep = ",");
ret <- as.xts(ret[, 2:ncol(ret)], order.by=as.Date(ret[,1]));
hmm <- regime_gmmhmm(ret, 1, 5)
#print(hmm$hmm$yhat_train);
#print(hmm$hmm$model)
#print(tail(hmm$stats, 200))
#print(ret[,0])
#stats = cbind(ret[,0], hmm$hmm$yhat);
#print(tail(stats))
