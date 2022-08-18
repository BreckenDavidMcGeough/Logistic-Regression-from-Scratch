from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error


df = pd.read_csv("diabetes.csv")



class LogisticRegression:
    def __init__(self,df):
        self.df = df
        self.alpha = 1e-2
        self.epochs = 1000
        self.x_train, self.x_test, self.y_train, self.y_test = self.preprocessing()
        self.weights = np.asmatrix([0 for _ in range(self.x_train.shape[1])]).transpose()
        
    def best_features(self):
        corr_matrix = self.df.corr()["Outcome"]
        best_matrix = []
        columns = self.df.columns
        for col in columns:
            if corr_matrix[col] > .2 and col != "Outcome":
                best_matrix.append(col)
        return best_matrix
    
    def preprocessing(self):
        best_features = self.best_features()
        self.df["Bias"] = [1 for _ in range(len(self.df))]
        best_features.insert(0,"Bias")
        X = self.df[best_features]
        X = scale(X)
        for i in range(X.shape[0]):
            X[i][0] = 1
        y = self.df[["Outcome"]]
        x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state = 5)
        x_train, x_test, y_train, y_test = np.asmatrix(x_train), np.asmatrix(x_test), np.asmatrix(y_train), np.asmatrix(y_test)
        return x_train,x_test,y_train,y_test
    
    def shapes(self):
        print("Shape x_train: " + str(self.x_train.shape))
        print("Shape x_test: " + str(self.x_test.shape))
        print("Shape y_train: " + str(self.y_train.shape))
        print("Shape y_test: " + str(self.y_test.shape))
        print("Shape weights: " + str(self.weights.shape))
        
    def activation(self,Z):
        return 1/(1+np.exp(-Z))
    
    def gradient(self):
        yHat = self.activation(np.dot(self.x_train,self.weights))
        gradJ = np.dot(self.x_train.transpose(),(yHat - self.y_train))
        return gradJ
    
    def gradient_descent(self):
        for _ in range(self.epochs):
            gradJ = self.gradient()
            self.weights = self.weights - self.alpha * gradJ
            
    def normalize_results(self):
        self.gradient_descent()
        predictions = self.activation(np.dot(self.x_test,self.weights))
        norm_predictions = []
        for pred in predictions:
            if pred >= .5:
                norm_predictions.append(1)
            else:
                norm_predictions.append(0)
        return norm_predictions
    
    def metrics(self):
        predictions = self.normalize_results()
        y = self.y_test
        num_wrong = 0
        for i in range(len(predictions)):
            if predictions[i] != y[i]:
                num_wrong += 1
        return ((len(predictions) - num_wrong)/len(predictions)) * 100
    
LGR = LogisticRegression(df)
print(LGR.metrics())
