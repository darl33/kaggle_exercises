
import pandas as pd

# Path of the file to read
file_path = 'C:/Users/darie/OneDrive/Documents/Desktop/stuf/code/intern@ntt/kaggle/kaggle_intro/melb_data.csv'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(file_path)



print(home_data.describe())