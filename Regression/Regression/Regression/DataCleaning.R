library(tidyr)
library(olsrr)

ori <- read.csv("BostonHousing_comma.csv", header=T)
head(ori)

x <- ori[,-c(1,15)]
y <- ori[, 15]

model <- lm(log(y) ~ 1 + x[,1] + x[,2] + x[,3] + x[,4] + x[,5] + x[,6] + x[,7] + x[,8] + x[,9] + x[,10] + x[,11] + x[,12] + x[,13])
k <- ols_step_all_possible(model)

colnames(x)
