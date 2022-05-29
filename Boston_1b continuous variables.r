##################################################################
#########  LocalGLMnet - Boston Housing analysis
#########  Author: Mario Wuthrich
#########  Version May 29, 2022
##################################################################
source("Boston_0a data.R")
source("Boston_0b networks.R")

path1 <- "../../LassoNet2/Plot/"

### for ron computer
#path1 <- "./LassoNet2/Plot/"

# data 
str(dat)


(y0 <- c(mean(learn$MEDV)))   # empirical means
###############################################
#########  fit regularized LocalGLMnet
###############################################

Q <- 18
features <- c(1:Q)

(q0 <- length(features))
Xlearn <- as.matrix(learn[, features])
Xlearn1 <- as.matrix(rep(1, nrow(learn)))
Ylearn <- as.matrix(learn$MEDV)

# set network parameters
(q00 <- c(q0, c(15,10)))
seed <- 200

# choose regularization path
eta0 <- c(0, 5, 10)
alpha0 <- c(0,1/2,1)

#### this fitting takes some time!
results <- array(NA, c(length(eta0), 1))

aa <- 3
epochs0 <- 300
   
for (jj in 1:length(eta0)){
    model <- LocalGLMnet.ElasticNet(seed, q00, y0[1], eta=eta0[jj], epsilon=10^(-5), alpha0[aa])
    path0 <- paste("./Networks/boston_eta",jj,"_alpha",aa, sep="")
    CBs <- callback_model_checkpoint(path0, monitor = "val_loss", verbose = 0,  save_best_only = TRUE, save_weights_only = TRUE)
    model %>% compile(loss = MSEP.reg, optimizer = 'nadam')
    fit <- model %>% fit(list(Xlearn, Xlearn1),  list(Ylearn),
                   validation_split=0.2, batch_size=5000, epochs=epochs0, verbose=1, callbacks=CBs)
    load_model_weights_hdf5(model, path0)
    learnP <- (model %>% predict(list(Xlearn, Xlearn1)))[,1]
    results[jj,1] <- round(c(square.loss(learnP, Ylearn)),4)
       }
  
results
##############################################
#########  analyze solutions
###############################################

aa <- 3

norm <- array(NA, c(length(eta0), q0))
results <- array(NA, c(length(eta0), 1))

for (jj in 1:length(eta0)){
    model <- LocalGLMnet.ElasticNet(seed, q00, y0[1], eta=eta0[jj], epsilon=10^(-5), alpha0[aa])
    path0 <- paste("./Networks/boston_eta",jj,"_alpha",aa, sep="")
    load_model_weights_hdf5(model, path0)
    w2 <- get_weights(model)
    learnP <- (model %>% predict(list(Xlearn, Xlearn1)))[,1]
    results[jj,1] <- round(c(square.loss(learnP, Ylearn)),4)
    zz <- keras_model(inputs=model$input, outputs=get_layer(model, 'Attention')$output)
    beta.x <- data.frame(zz %>% predict(list(Xlearn, Xlearn1)))
    norm[jj,] <- colSums(abs(beta.x))/nrow(beta.x)
     }

#norm
results

col0 <- rev(rainbow(n=length(eta0), start=0, end=.3))

norm0 <- norm

for (jj in rev(1:(nrow(norm0)-1))){norm0[jj,] <- pmax(0,norm[jj,] - norm[jj+1,])}

norm0 <- norm0[rev(1:nrow(norm0)),]
norm0 <- norm0[,rev(1:ncol(norm0))]

plot.yes <- 1
plot.yes <- 0
filey1 <- paste(path1, "Importance_Boston1.pdf", sep="")
if (plot.yes==1){pdf(file=filey1)}
par(mar=c(2,5,4,2)) # increase y-axis margin.
barplot(t(t(norm0)), las=1, col=rev(col0), cex.lab=1.5, cex.axis=1.5, cex.names=1.2, names.arg=rev(names(learn)[features]), main=list(paste("importance measure (alpha=",alpha0[aa],")", sep=""), cex=1.5), horiz=TRUE)
abline(v=0)
abline(v=c(0:6)/10, lty=2, col="darkgray")
#abline(v=mean(norm[1,7:8]), col="black", lwd=2)
legend(x="bottomright", bg="white", cex=1.5, fill=col0, legend=paste("eta=",eta0, sep=""))
if (plot.yes==1){dev.off()}

############################################# refit - all real vars 
Q <- 13
features0 <- c(1:Q)

results <- array(NA, dim=c(Q))

(q0 <- length(features))
Xlearn <- as.matrix(learn[, features])
Xlearn1 <- as.matrix(rep(1, nrow(learn)))
Ylearn <- as.matrix(learn$MEDV)

(q00 <- c(q0, c(15,10)))

epochs0 <- 300
model <- LocalGLMnet.ElasticNet(seed=100, q00, y0[1], eta=0, epsilon=10^(-5), 1)
path0 <- paste("./Networks/Boston_cont_part2", sep="")
CBs <- callback_model_checkpoint(path0, monitor = "val_loss", verbose = 0,  save_best_only = TRUE, save_weights_only = TRUE)
model %>% compile(loss = MSEP.reg, optimizer = 'nadam')

fit <- model %>% fit(list(Xlearn, Xlearn1),  list(Ylearn), 
                     validation_split=0.2, batch_size=5000, epochs=epochs0, verbose=0, callbacks=CBs)

load_model_weights_hdf5(model, path0)
learnP <- (model %>% predict(list(Xlearn, Xlearn1)))[,1]
results<- c(round(c(square.loss(learnP, Ylearn)),4))