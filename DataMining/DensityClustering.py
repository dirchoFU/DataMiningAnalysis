import pandas as pd
from numpy import int64, array
import math
import csv

iris = pd.read_csv('iris.txt', sep=',', encoding='utf-8')
iris.columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
#g=sns.FacetGrid(iris,hue='Species')
#g.set(xlim=(0,2.5),ylim=(0,7))
#g.map(plt.scatter,'PetalWidthCm','PetalLengthCm').add_legend()


X=iris[['PetalWidthCm','PetalLengthCm']]
from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=0.3,min_samples=10)
dbscan.fit(X)
dbscan.labels_
array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,
        1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1, -1,  1,
        1, -1,  1, -1,  1, -1, -1, -1,  1,  1,  1,  1, -1,  1,  1, -1, -1,
        1,  1,  1, -1,  1,  1, -1,  1,  1,  1, -1, -1, -1,  1,  1, -1, -1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1], dtype=int64)
from sklearn import datasets
from pandas import DataFrame
noisy_circles=datasets.make_circles(n_samples=1000,factor=.5,noise=.05)
print(noisy_circles)
df=DataFrame()
df['x1']=noisy_circles[0][:,0]
df['x2']=noisy_circles[0][:,1]
df['label']=noisy_circles[1]
df.sample(10)

dbscan=DBSCAN(eps=0.2,min_samples=10)
X=df[['x1','x2']]
dbscan.fit(X)
df['dbscan_label']=dbscan.labels_
g=sns.FacetGrid(df,hue='dbscan_label')
g.map(plt.scatter,'x1','x2').add_legend()
plt.show()

class flower:
    def __init__(self, sepal_l, sepal_w, petal_l, petal_w, type, group="not-visited"):
        self.sepal_l = sepal_l
        self.sepal_w = sepal_w
        self.petal_l = petal_l
        self.petal_w = petal_w
        self.type = type
        self.group = group


# 基于密度的聚类分析算法，eps是扫描半径，MinPts是最小包含点数
def dbscan(dataset, eps, MinPts):
    c = 0

    for item in dataset:
        if item.group != "not-visited":
            continue

        item.group = "visited"
        NeighborPts = regionQuery(item, eps, dataset)

        if (len(NeighborPts) < MinPts):
            item.group = "outlier"
        else:
            c = c + 1
            expandCluster(item, NeighborPts, c, eps, MinPts, dataset)

    return c


# 对簇进行扩展
def expandCluster(item, NeighborPts, c, eps, MinPts, dataset):
    item.group = c
    for itens in NeighborPts:
        if itens.group == "not-visited":
            itens.group = "visited"
            NeighborPts_ = regionQuery(itens, eps, dataset)
            if (len(NeighborPts_) >= MinPts):
                NeighborPts.union(NeighborPts_)
            if itens.group != "not-visited":
                itens.group = c


def regionQuery(item, eps, dataset):
    NeighborPts = set()
    for itens in dataset:
        if (euclidian(item, itens) <= eps):
            NeighborPts.add(itens)
    return NeighborPts


def euclidian(subject1, subject2):
    soma = pow(subject1.sepal_l - subject2.sepal_l, 2) + \
           pow(subject1.sepal_w - subject2.sepal_w, 2) + \
           pow(subject1.petal_l - subject2.petal_l, 2) + \
           pow(subject1.petal_w - subject2.petal_w, 2)
    result = math.sqrt(soma)
    return result


def main():
    dataset = list()
    radius = 3
    density = 4

    with open('iris.csv', 'r') as csvfile:
        lines = csv.reader(csvfile)
        for row in lines:
            dataset.append(flower(float(row[0]), float(row[1]), float(row[2]), float(row[3]), row[4]))

    clusters = dbscan(dataset, radius, density)

    print("半径: {}".format(radius))
    print("密度: {}".format(density))
    print("生成的聚类数量: {}".format(clusters))


if __name__ == '__main__':
    main()