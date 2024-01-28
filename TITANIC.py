from sklearn.model_selection import train_test_split
import pandas as pd
import math
import matplotlib
from future.moves import  tkinter
import numpy as np
#import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
#url = "https://drive.google.com/drive/folders/1rNNROo4PIqtiKAAZAT5D8M2E9q5xBXG6"

file = open("train.csv", "r")
data = pd.read_csv("train.csv", sep=",")
data2 = pd.read_csv("test.csv",sep=",")
data3 = pd.read_csv("gender_submission.csv",sep=",")
plt.interactive(False)
matplotlib.use('TkAgg')
data.dropna()
data2.dropna()

#print(data.head())
#print(data.info())

#print("data2")
#print(data2.head())
#print(data2.info())

data['topic_id'] = pd.factorize(data.Sex)[0]
data2['topic_id'] = pd.factorize(data2.Sex)[0]

#data.dropna(subset=['Age'])
#np.all(np.isfinite(data))
#np.any(np.isnan(data))

X_train = data[['Pclass','Fare','topic_id','Parch','SibSp']]
y_train = data[['Survived']]

X_test = data2[['Pclass','Fare','topic_id','Parch','SibSp']]
y_test = data3[['Survived']]



#y_train=np.reshape(y_train, (891,1))
#print("y_train shape")
#print(y_train.shape)
#from numpy import genfromtxt
#my_data = genfromtxt('train.csv', delimiter=',')
#print(my_data)

#data['Sex'].astype('category').cat.codes
#data[‘Sex’] = (data[‘Gender’] ==‘Male’).astpye(int)
#gender = {'male': 1,'female': 0}
# #data.Sex = [gender[item] for item in data.Sex]
#data.Sex[data.Sex == 'male'] = 1
#data.Sex[data.Sex == 'female'] = 0
#y = y.values.reshape(535,1)



def safe_ln(x, minval=0.0000000001):     #avoid divide by zero
    return np.log(x.clip(min=minval))

def normalize(X):
    max=np.max(X,axis=0)

    normX= X/max
    #print("Norm X")
    #print(normX)
    return normX

def logisticFunc(beta,X):
    #sigmoid
    return 1.0/(1 + np.exp(-np.dot(X,beta.T)))

def lgradient(beta, X,y_train):    #function tp give gradient to subtract
    f= logisticFunc(beta,X)
    #print("f shape 1")
    #print(f.shape)
    f =f - y_train
    #print("f shape 1.5")
    #print(f.shape)
    f = np.dot(f.T,X)
    #print("f shape 2")
    #print(f.shape)
    return f

def cost_func(beta, X,y_train):
    v=logisticFunc(beta,X)
    #print("v shape")
    #print(v.shape)
    #print("y_train shape")
    #print(y_train.shape)
    np.reshape(y_train,(891,1))
    final=-(y_train*np.log(v))-((1-y_train)*safe_ln(1-v))
    return np.mean(final)

def grad_desc(X,y_train,beta, lr=.001):
    cost = cost_func(beta,X,y_train)
    num_iter=1
    cost_record = []
    while(  num_iter<1000 ):

        beta = beta - (lr*lgradient(beta,X,y_train))
        cost = cost_func(beta, X, y_train)
        #print("cost:")
        #print(cost)

        num_iter +=1
        cost_record.append([cost, num_iter])

    cost_record = np.array(cost_record)

    return beta,num_iter,cost_record

def pred_value(beta, X):
    pred_prob = logisticFunc(beta,X)

    #print("pred_prob is:")
    #print(pred_prob)
    #print("shape of pred_prob:")
    #print(pred_prob.shape)
    pred = np.where(pred_prob >= .5,1,0)
    return pred


def plot(cost_record):
    Y= cost_record[:,0]
    X=cost_record[:,1]

    plt.scatter(X, Y)
    plt.xlabel('entry X')
    plt.ylabel('entry Y')
    plt.show()

if __name__=="__main__":

    X = normalize(X_train)
    X =   np.hstack((np.matrix(np.ones(X_train.shape[0])).T, X_train)) # 0 means along a column
    #initial beta values


    beta = np.matrix(np.zeros(X.shape[1]))
    #print("beta is:")
    #print(beta)


    beta, num_iter,cost_record = grad_desc(X, y_train, beta)
    
    np.savetxt('beta.txt', beta)
    
    print("beta is:")
    print(beta)
    print(beta.shape)
    
    print("X_test.shape")
    print(X_test.shape)
    print(X_test.head())
    print(X_test.info())

    X_test = np.hstack((np.matrix(np.ones(X_test.shape[0])).T, X_test))
    
    print("X_test.shape")
    print(X_test.shape)
    #print(X_test.head())
    print(X_test)
    y_final = pred_value(beta,X_test)
    print("Correct predictions:",np.sum(y_test== y_final))

    #plot(cost_record)


