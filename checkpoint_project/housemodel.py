import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd


fpath = r'C:\Users\darie\OneDrive\Documents\Desktop\stuf\code\intern@ntt\kaggle\checkpoint_project\data.csv'
home_data = pd.read_csv(fpath)
y = home_data.price
home_data['date'] = pd.to_datetime(home_data['date'])
home_data['year'] = home_data['date'].dt.year
home_data['month'] = home_data['date'].dt.month
home_data['day'] = home_data['date'].dt.day

home_data[['zipcode', 'statezip']] = home_data['statezip'].str.split(' ', n=1, expand=True)

# using label encoder for strings-> narrow down data
le = preprocessing.LabelEncoder()


home_data['city'] = le.fit_transform(home_data['city'])
home_data['country'] = le.fit_transform(home_data['country'])
home_data['street'] = le.fit_transform(home_data['street'])
home_data['statezip'] = le.fit_transform(home_data['statezip'])

features = [
'year', 'month', 'day',
'price','bedrooms','bathrooms','sqft_living','sqft_lot',
'floors','waterfront','view','condition',
'sqft_above','sqft_basement','yr_built',
'yr_renovated','street',
'city',
'statezip',
'country'
]

# Select columns corresponding to features, and preview the data
X = home_data[features]
X.head()

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define a random forest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
rf_model_on_full_data = RandomForestRegressor(random_state=1)

rf_model_on_full_data.fit(X,y)
# path to file you will use for predictions
test_data_path = fpath

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

test_X = test_data[features]

# make predictions. 
test_preds = rf_model_on_full_data.predict(test_X)
full_mae = mean_absolute_error(test_preds, y)
print("mae for full data: {:,.0f}".format(full_mae))
# output = pd.DataFrame({'Id': test_data.Id,
#                        'SalePrice': test_preds})
# output.to_csv('submission.csv', index=False)