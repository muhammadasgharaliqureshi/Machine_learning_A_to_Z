setwd('E:\\asghar\\Machine\ learning\\Course\ learninig\ coding')

data_set = read.csv('Data.csv')

data_set$Age = ifelse(is.na(data_set$Age), ave(data_set$Age), FUN = function(x) mean(x, na.rm(TRUE)), data_set$Age) 


data_set$Salary = ifelse(is.na(data_set$Salary), ave(data_set$Salary), FUN = function(x) mean(x, na.rm(TRUE)), data_set$Salary)


