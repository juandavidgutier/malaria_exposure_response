library(EValue)

#for rainfall
param_evalue_rain <- read.csv("D:/param_evalue_rain.csv")

param_evalue_rain$EValue <- 0

evalue_current_rain <- evalues.MD(est = param_evalue_rain$Cohen_s_d[1], se = param_evalue_rain$SE[1])
param_evalue_rain[1,7] <- evalue_current_rain[2,1]

evalue_avg2_rain <- evalues.MD(est = param_evalue_rain$Cohen_s_d[2], se = param_evalue_rain$SE[2])
param_evalue_rain[2,7] <- evalue_avg2_rain[2,1]

print(param_evalue_rain)




#for temperature
param_evalue_temp <- read.csv("D:/param_evalue_temp.csv")

param_evalue_temp$EValue <- 0

evalue_current_temp <- evalues.MD(est = param_evalue_temp$Cohen_s_d[1], se = param_evalue_temp$SE[1])
param_evalue_temp[1,7] <- evalue_current_temp[2,1]

evalue_avg2_temp <- evalues.MD(est = param_evalue_temp$Cohen_s_d[2], se = param_evalue_temp$SE[2])
param_evalue_temp[2,7] <- evalue_avg2_temp[2,1]

print(param_evalue_temp)
