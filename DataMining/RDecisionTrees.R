library(rpart)
library(rpart.plot)
data4<-read.table("iris.txt",sep = ",")
summary(data4)
ind<-sample(2,nrow(data4),replace = TRUE,prob = c(0.7,0.3))
train<-data4[ind==1,]
test<-data4[ind==2,]
tree<-rpart(data4[1:2],data=train,method = "class")

rpart.plot(tree)
rpart.plot(tree,branch=1,branch.type=2,type = 1,extra = 102,shadow.col = "gray",
           box.col="green",border.col="blue",split.col="red",split.cex=1.2,
           main="决策树")
printcp(tree)
