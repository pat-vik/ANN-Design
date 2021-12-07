import pandas as pd

# read the LBW_Dataset.csv file
data=pd.read_csv('LBW_Dataset.csv')                             

# Shuffle the rows of the dataframe
data = data.sample(frac=1).reset_index(drop=True)              

# Handling of missing values

 # Age is filled with the mean
data['Age'] = data['Age'].fillna(round(data['Age'].mean()))    

# By just sorting the dataset according to the weights, we observed that for lower values of weights(30 to 35) the 'Result' was mostly 0
# We observed that for all the missing values for weights the 'Result' column had value 0
# so Weight is filled with 31, which is a number we chose between 30 to 35.
data['Weight'] = data['Weight'].fillna(31)                       

# Delivery phase is filled with the mode which is 1 
data['Delivery phase'] = data['Delivery phase'].fillna(1.0)     

# HB is filled with the mean rounded off to one decimal place 
data['HB'] = data['HB'].fillna(round(data['HB'].mean(),1))      

# BP is filled with the mean
data['BP'] = data['BP'].fillna(data['BP'].mean())               

# Residence is filled with the mode which is 1 
data['Residence'] = data['Residence'].fillna(1.0)               

# Education column is removed as it has only 5 as its values 
data = data.drop(columns=['Education'])                         

# write the modified dataset to preprocessed_data.csv
data.to_csv('preprocessed_data.csv',index = False)              