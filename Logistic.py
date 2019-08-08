from numpy import *
from math import *


def loaddata():
    dataMat=[]
    labelMat=[]
    fp=open("data.txt")
    for line in fp.readlines():
        lineArr=line.strip().split("   ")
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat
def sigmoid(x):
    return 1.0/(1+exp(-x))
def gradAscent(dataMatIn,labelMatIn):
    dataMatrix=mat(dataMatIn)
    labelMatrix=mat(labelMatIn).transpose()
    m,n=shape(dataMatrix)
    alpha=0.001
    cycles=500

    weight=ones((n,1))
    for i in range(0,cycles):
        temp = []
        x=dataMatrix*weight
        for j in range(0,100):
            h=sigmoid(x[j])
            temp.append(h)
        matrix=mat(temp).transpose()
        error=labelMatrix-matrix
        weight=weight+alpha*dataMatrix.transpose()*error
    return weight
def plotFit(w,dataMatrix,labelMatrix):
    import matplotlib.pyplot as plt
    weight=w.getA()
    dataArr=array(dataMatrix)
    n=shape(dataMatrix)[0]
    xcord1=[]
    ycord1=[]
    xcord2=[]
    ycord2=[]
    for i in range(n):
        if int(labelMatrix[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-3.0,3.0,0.1)
    y=(-weight[0]-weight[1]*x)/weight[2]
    ax.plot(x,y)
    #plt.xlable('x1')
    #plt.ylable('x2')
    plt.show()
if __name__=='__main__':
    dataMatrix,labelMatrix=loaddata()
    weight=gradAscent(dataMatrix,labelMatrix)
    plotFit(weight,dataMatrix,labelMatrix)