import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import missingno as msno # visualizing missing values
import seaborn as sns # for data visualization
import matplotlib.pyplot as plt # for plot

import torch
import torch.nn as nn # For neural network building
import torch.nn.functional as F # For activation function
from torch.utils import data  # For data handling
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler # Same as previous
from sklearn.model_selection import train_test_split # For splitting the data
from sklearn.model_selection import KFold # For kfold training
import os

df_train_raw=pd.read_csv('/Users/Apple/Downloads/house-prices-advanced-regression-techniques/train.csv')
#sns.histplot(df_train_raw['SalePrice'], kde=True)
#plt.show()

def preprocess(df):
    df=df.copy()
    df=df.drop(columns=["Id"],errors='ignore')
    df_numerical=df.select_dtypes(np.number)
    df_numerical=df_numerical.drop( columns="SalePrices",errors="ignore")
    df_no_numerical=df.select_dtypes(include='object')
    df_no_numerical=pd.get_dummies(df_no_numerical,dummy_na=True)
    df_numerical=df_numerical.apply(lambda x: (x-x.mean())/(x.std()))
    df=pd.concat([df_numerical,df_no_numerical],axis=1)
    df=df.fillna(0)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    return df

df_train_y=df_train_raw["SalePrice"].copy()
df_train_y=np.log(df_train_y,where=df_train_y!=0)
df_train_x=preprocess(df_train_raw)
df_train_x,df_val_x,df_train_y,df_val_y=train_test_split(df_train_x,df_train_y,train_size=0.8,random_state=42)
class FNN(nn.Module):
    def __init__(self,d_in=332,d_out=1,n=300,hn=5):
        super().__init__()
        self.activation=nn.Softplus()
        self.layers=nn.ModuleList([nn.Linear(d_in,n),self.activation])
        for i in range(1,hn-1):
            self.layers.extend([nn.Linear(n,n),self.activation])
        self.layers.append(nn.Linear(n,d_out))
    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        return x
tensor_val_y=torch.tensor(df_val_y.values).float()
tensor_val_y=tensor_val_y.reshape(-1,1)
tensor_val_x=torch.tensor(df_val_x.values).float()
tensor_train_x=torch.tensor(df_train_x.values).float()
tensor_train_y=torch.tensor(df_train_y.values).float()
tensor_train_y=tensor_train_y.reshape(-1,1)
dataset=TensorDataset(tensor_train_x,tensor_train_y)
dataloader=DataLoader(dataset,batch_size=16,shuffle=True)
model=FNN()
optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=0.001)
loss_fn=nn.MSELoss()
epochs=500
for epoch in range(epochs):
    for batch,(x,y) in enumerate(dataloader):
        tensor_predict_y=model(x)
        loss=loss_fn(tensor_predict_y,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

numpy_predict_y = tensor_predict_y.detach().numpy()
df_predict_y=pd.DataFrame(numpy_predict_y,columns=["prediction"])
fig,ax1=plt.subplots()
sns.histplot(df_predict_y["prediction"],kde=True,color="blue",ax=ax1,bins=10)
ax1.set_title("prediction")
fig,ax2=plt.subplots()
sns.histplot(df_train_raw['SalePrice'], kde=True,color="green",ax=ax2)
ax2.set_title("real")

plt.show()
