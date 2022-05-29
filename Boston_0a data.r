##################################################################
#########  LocalGLMnet - Boston Housing data processing
#########  Author: Ronald Richman & Mario Wuthrich
#########  Version May 29, 2022
##################################################################
#1    CRIM - per capita crime rate by town
#2    ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
#3    INDUS - proportion of non-retail business acres per town.
#4    CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
#5    NOX - nitric oxides concentration (parts per 10 million)
#6    RM - average number of rooms per dwelling
#7    AGE - proportion of owner-occupied units built prior to 1940
#8    DIS - weighted distances to five Boston employment centres
#9    RAD - index of accessibility to radial highways
#10    TAX - full-value property-tax rate per $10,000
#11    PTRATIO - pupil-teacher ratio by town
#12    B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#13    LSTAT - % lower status of the population
#    MEDV - Median value of owner-occupied homes in $1000's
##########################################
#########  data pre-processing
##########################################

dat <- read.table(file="HousingData.csv", header=TRUE, sep=",")

J0 <- 13
mm <- colMeans(dat[,1:J0], na.rm=TRUE)
for (j in 1:J0){dat[is.na(dat[,j]),j] <- mm[j]}

set.seed <- 100
dat$X1 <- dat[sample(nrow(dat)),"CRIM"] 
dat$X2 <- dat[sample(nrow(dat)),"CHAS"] 
dat$X3 <- dat[sample(nrow(dat)),"RM"] 
dat$X4 <- dat[sample(nrow(dat)),"RAD"] 
dat$X5 <- dat[sample(nrow(dat)),"LSTAT"] 
dat <- cbind(dat[,1:J0], dat[,(J0+2):(J0+6)], dat[,J0+1])   
names(dat)[J0+6] <- "MEDV"                                   

for (j in 1:(J0+5)){dat[,j] <- (dat[,j]-mean(dat[,j]))/sd(dat[,j])}

learn <- dat#[!ll,]