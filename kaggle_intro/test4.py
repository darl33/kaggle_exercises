
import pandas as pd
import random
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


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
mean_absolute_error(y, predictions)

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
