 library(ggplot2)
 data <- read.table("magic04.txt",sep=",",header=F)
 #多元均值向量计算
 datem<-as.matrix(data[,1:10])
 dm<-matrix(nrow=1,ncol = 10)
 for (i in 1:10) { dm[,i]=mean(data[,i]) }
 print("各属性均值如下：")
 dm
 #中心数据矩阵
 dt<-matrix(nrow = 19020,ncol = 10)
 for (i in 1:10) {dt[,i]=datem[,i]-1*dm[,i]}  
 t_dt<-matrix(nrow = 10,ncol = 19020)
 t_dt=t(dt)
 
 inner=t_dt%*%dt*(1/19020)
 inner
 #外积样本协方差矩阵
 tcrossprod(inner)

 #相关性散点图
 cosdata<-crossprod(dt[,1],dt[,2])/
   (sqrt(sum(dt[,1]^2))*sqrt(sum(dt[,2]^2)))
 ggplot(data[,1:2], aes(x = data[,1], y = data[,2]))+
   geom_point(col="blue")+geom_smooth(method = lm)
 cosdata
 #正态分布概率密度图
 hist(data[,1],freq=FALSE)
 lines(density(data[,1]),col="blue") 
 #variance计算与计较
 c=array()
 for (i in 1:10) { c[i]=var(data[,i]) }
 for (j in 1:10) { if(c[j]==max(c)) print(paste("属性",j,"方差最大为",max(c))) 
   if(c[j]==min(c)) print(paste("属性",j,"方差最小为",min(c)))}
 #covariance计算与计较
 d <- matrix(nrow = 10,ncol = 10)
 for (i in 1:10) { for(j in 1:10) d[i,j]=cov(data[,i],data[,j]) }
 for (i in 1:10) { for(j in 1:10) if(d[i,j]==max(d,na.rm = TRUE))
 {print(paste("属性",i,"和",j,"协方差最大为",max(d,na.rm = TRUE)))}}
 for (i in 1:10) { for(j in 1:10) if(d[i,j]==min(d,na.rm = TRUE))
 {print(paste("属性",i,"和",j,"协方差最小为",min(d,na.rm = TRUE)))}}


