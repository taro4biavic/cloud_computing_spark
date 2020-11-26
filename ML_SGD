import pandas as pd
import numpy as np
import random
import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

train = np.loadtxt('/Users/zhenghaoyu/Downloads/data_banknote_authentication.txt',delimiter=',')

from sklearn.metrics import mean_squared_error
xtrain = train[:,:-1]
ytrain = train[:,-1]
inv = np.linalg.pinv(xtrain)
wlin = np.dot(inv,ytrain)
yhat = np.dot(xtrain,wlin)
elin = mean_squared_error(ytrain,yhat)


iters = 1000
eta = 0.001
res = 0
for i in range(iters):
  w = np.zeros(4)
  random.seed(i)
  k=0
  while k<500:
    ran = math.floor(random.random()*len(xtrain))
    gradient = sigmoid(ytrain[ran]*np.dot(w.T,xtrain[ran])*(-1)) * (ytrain[ran]*xtrain[ran])
    w = w + eta * gradient
    k+=1
  yhat = np.dot(xtrain,w)
  ecein = np.sum(np.log(1+np.exp((-1)*np.multiply(ytrain,yhat))))/len(ytrain)
  res+=ecein
print(res/iters)
