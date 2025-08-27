import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd

fpath = r'C:\Users\darie\OneDrive\Documents\Desktop\stuf\code\intern@ntt\kaggle\checkpoint_project\data.csv'
with open(fpath, 'r') as file:
    for _ in range(15):
        print(file.readline())
home_data = pd.read_csv(fpath)
print(home_data.dtypes)