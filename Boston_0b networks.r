##################################################################
#########  LocalGLMnet - Boston Housing network functions
#########  Author: Ronald Richman & Mario Wuthrich
#########  Version May 29, 2022
##################################################################

library(keras)

##########################################
#########  help functions
##########################################
 
square.loss <- function(pred, obs){mean((pred-obs)^2)}

MSEP.reg <- function(y_true, y_pred){k_mean((y_true[,1]-y_pred[,1])^2 + y_pred[,2])}

plot.loss.title <- function(pos0, loss, title0, ylim0, plot.yes=0, filey1, col0){
  if (plot.yes==1){pdf(filey1)}      
    plot(loss$val_loss, col=col0[2], ylim=ylim0, main=list(title0, cex=2),xlab="training epochs", ylab="(modified) deviance loss", cex=1.5, cex.lab=1.5)
    lines(loss$loss,col=col0[1])
    abline(v=which.min(fit[[2]]$val_loss), col=col0[3])
    legend(x=pos0, cex=1.5, col=col0, lty=c(1,-1,1), lwd=c(1,-1,1), pch=c(-1,1,-1), legend=c("training loss", "validation loss", "minimal validation loss"))
if (plot.yes==1){dev.off()}          
   }

###############################################
#########  networks
###############################################

LocalGLMnet.ElasticNet <- function(seed, q00, y0, eta, epsilon, alpha){
    set.seed(seed)
    tensorflow::set_random_seed(seed)
    Design  <- layer_input(shape = c(q00[1]), dtype = 'float32', name = 'Design') 
    Bias1  <- layer_input(shape = c(1), dtype = 'float32', name = 'Bias1')
    #
    Attention = Design %>% 
          layer_dense(units=q00[2], activation='tanh', name='FNLayer1') %>%
          layer_dense(units=q00[3], activation='tanh', name='FNLayer2') %>%
          layer_dense(units=q00[1], activation='linear', name='Attention', 
                  weights=list(array(0, dim=c(q00[3],q00[1])), array(0,dim=c(q00[1]))))
    #         
    Penalty0 = Attention %>% layer_lambda(function(x) k_square(x))
    Penalty1 = Penalty0 %>%
              layer_lambda(function(x) eta*alpha*k_sqrt(x+epsilon)) %>%
              layer_dense(units=1, activation='linear', name='Penalty1', 
                    weights=list(array(1, dim=c(q00[1],1))), use_bias=FALSE, trainable=FALSE)
    Penalty2 = Penalty0 %>%
              layer_lambda(function(x) eta*(1-alpha)*x) %>%
              layer_dense(units=1, activation='linear', name='Penalty2', 
                    weights=list(array(1, dim=c(q00[1],1))), use_bias=FALSE, trainable=FALSE)
    Penalty = list(Penalty1,Penalty2) %>% layer_add()
    #
    LocalGLM = list(Design, Attention) %>% layer_dot(name='LocalGLM', axes=1)
    #
    Bias = Bias1 %>%
           layer_dense(units=1, activation='linear', name='Bias',
                    weights=list(array(y0, dim=c(1,1))), use_bias=FALSE)
    #
    Response = list(LocalGLM, Bias) %>% layer_add()
    Output = list(Response, Penalty) %>% layer_concatenate()
    #
    keras_model(inputs = c(Design, Bias1), outputs = c(Output))
    }
