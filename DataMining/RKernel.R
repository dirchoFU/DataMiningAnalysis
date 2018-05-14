library(lattice)
iris<-read.table("iris.txt",sep = ",")
names(iris)=c("Sepal.Length","Sepal.Width","Petal.Length","Petal.Width","Species")
#  Gaussian Kernal
#  高斯核
kernalGaussian <- function(xData)
{
  #  得到相应的核函数
  #if(ncol(xData)!=1){ stop('error input data') }
  
  stdX <- sd(xData)
  #  高斯宽带的选择
  h <- 1.06*stdX*length(xData)^(-1/5) 
  kernalX <- 1/(h*sqrt(2*pi)) * exp(-xData^2/(2*h^2))
  return(kernalX)
}
kernalGaussian(iris[,4])
plot(kernalGaussian(iris[,4]))
xyplot(iris[,3] ~ iris[,4], data = iris, groups = iris[,5],
       auto.key=list(corner=c(1,0)))
#中心化
mean_i<-matrix(ncol=4)
for(i in 1:4){mean_i[,i]=mean(iris[,i])}
mean_iris<-matrix(nrow = 150,ncol=4)
for(i in 1:4){mean_iris[,i]=iris[,i]-mean_i[,i]}
#归一化
autonorm<-function(data){
  min<-min(data)
  max<-max(data)
  for(i in 1:length(data))
    data[i]<-(data[i]-min)/(max-min)
  return(data)
}
dataRK<-apply(mean_iris[,1:4],2,autonorm)
data_k<-dataRK^2
#验证归一化
x<-iris[13,1:4]
y<-iris[79,1:4]
z<-iris[100,1:4]
x<-(x-apply(iris[c(-13,-79,-100),1:4],2,min))/(apply(iris[c(-13,-79,-100),1:4],2,max)-apply(iris[c(-13,-79,-100),1:4],2,min))
y<-(y-apply(iris[c(-13,-79,-100),1:4],2,min))/(apply(iris[c(-13,-79,-100),1:4],2,max)-apply(iris[c(-13,-79,-100),1:4],2,min))
z<-(z-apply(iris[c(-13,-79,-100),1:4],2,min))/(apply(iris[c(-13,-79,-100),1:4],2,max)-apply(iris[c(-13,-79,-100),1:4],2,min))
#核矩阵
data_kernel=data_k%*%t(data_k)
plot(data_kernel)
