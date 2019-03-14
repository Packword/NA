import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron as skPerceptron
from sklearn.metrics import accuracy_score
def loss(y_pred,y):
    y_pred = y_pred.reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    return ((y_pred-y)**2).mean()/2
class Perceptron:
    def __init__(self, W=None,b=0):
        self.W=W
        self.b=b
    def activate(self, x):
        return np.array(x > 0, dtype=np.int64)
    def forward_pass(self,X):
        n=X.shape[0]
        y_pred=np.zeros((n,1))
        y_pred = self.activate(X @ self.W.reshape(X.shape[1], 1) + self.b)
        return y_pred.reshape(-1, 1)
    def backward_pass(self, X, y, y_pred, learning_rate=0.005):
        n = len(y)
        y = np.array(y).reshape(-1, 1)
        delta_w= learning_rate * (X.T @ (y_pred - y) / n)
        delta_b= np.mean(y_pred - y)
        self.W -= delta_w
        self.b -= delta_b
    def fit(self,X,y,num_iterac=10000):
        self.W=np.zeros((X.shape[1],1))
        self.b=0
        losses=[]
        for i in range(num_iterac):
            y_pred=self.forward_pass(X)
            losses.append(loss(y_pred,y))
            self.backward_pass(X, y, y_pred)
        return losses
data=pd.read_csv('voice.csv')
data['label'] = data['label'].apply(lambda x: 1 if x == 'male' else 0)
data.head()
data = data.sample(frac=1)
X_train = data.iloc[:int(len(data)*0.7), :-1]
y_train = data.iloc[:int(len(data)*0.7), -1]
X_test = data.iloc[int(len(data)*0.7):, :-1]
y_test = data.iloc[int(len(data)*0.7):, -1]
sk_perceptron=skPerceptron(random_state=42)
sk_perceptron.fit(X_train,y_train)
perceptron=Perceptron()
perceptron.fit(X_train.values,y_train.values)
print('Точность (доля правильных ответов, из 100%) моего перцептрона: {:.1f} %'
      .format(accuracy_score(y_test, perceptron.forward_pass(X_test)) * 100))
print('Точность (доля правильных ответов) перцептрона из sklearn: {:.1f} %'
      .format(accuracy_score(y_test, sk_perceptron.predict(X_test)) * 100))
