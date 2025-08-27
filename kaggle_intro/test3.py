
import pandas as pd
import random
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error



# Path of the file to read
file_path = 'C:/Users/darie/OneDrive/Documents/Desktop/stuf/code/intern@ntt/kaggle/kaggle_intro/melb_data.csv'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(file_path)
y = home_data.Price
feature_names = ['LotArea',
'YearBuilt'
,'1stFlrSF'
,'2ndFlrSF'
,'FullBath'
,'BedroomAbvGr'
,'TotRmsAbvGrd']

# Select data corresponding to features in feature_names
X = home_data[feature_names]

#specify the model. 
#For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = DecisionTreeRegressor(random_state = random.randomint(1,10))

# Fit the model
iowa_model.fit(X, y)

print(home_data.columns())

home_data.dropna(axis=0)
y = home_data.price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = home_data[melbourne_features]
print(X.describe())


print(home_data.describe())

predictions = iowa_model.predict(X)
print(predictions)
