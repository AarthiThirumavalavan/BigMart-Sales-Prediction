# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 09:38:26 2018

@author: thirumav
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 10:21:41 2018

@author: thirumav
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
 
#STEP1: LOADING DATA
#Reading the train and test files
train_data = pd.read_csv("C:/Users/thirumav/Desktop/Python_exercises/Big Mart/Train_data.csv")
test_data = pd.read_csv("C:/Users/thirumav/Desktop/Python_exercises/Big Mart/Test_data.csv")

#STEP2: DATA EXPLORATION
#Combining train and test files
train_data['source'] = "train"
test_data['source'] = "test"
data = pd.concat([train_data, test_data])
print("Train data size: ", train_data.shape)
print("Test data size: ", test_data.shape)
print("Concatenated data size: ", data.shape)
#To find which columns contain null values
null_counts = data.isnull().sum()
print(null_counts)
#To find the number of unique values in each categorical field
data.apply(lambda x : (len(x.unique())))
#Filter the categorical fields
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x] == 'object']
#Removing the ID fields since they do not contribute to the prediction. Also Source(train/test) is removed.
categorical_columns = [x for x in categorical_columns if x not in ('Item_Identifier', 'Outlet_Identifier', 'source')]
#To count the number of categories in each column
for col in categorical_columns:
    print("\nFrequency of categories for column: ", col)
    print(data[col].value_counts())

#STEP3: DATA CLEANING
#Treat Item_Weight for missing values
#Replace the missing values by the average of Item_Weight under each group
data["Item_Weight"] = data.groupby("Item_Identifier").Item_Weight.transform(lambda x: x.fillna(x.mean()))
#Replace missing values of Outlet_Size by the mode under each Outlet_Type since it is a categorical field
data["Outlet_Size"] = data.groupby("Outlet_Type").Outlet_Size.transform(lambda x: x.fillna(x.mode()[0]))

#STEP4: FEATURE ENGINEERING
#Since Item_Visibility cannot be 0, we replace 0 by mean visibility corresponding to Item_Identifier
data["Item_Visibility"] = data.groupby("Item_Identifier").transform(lambda x: x.fillna(x.mean()))
#Since Item_Type falls under three categories of FD:Food, DR: Drink, NC: Non-consumable based on the Item_Identifier
#the Item_Type is changed to fit into one of the three categories
data["Item_Categoric_Type"] = data["Item_Identifier"].apply(lambda x: x[0:2]) 
data["Item_Categoric_Type"] = data["Item_Categoric_Type"].map({"FD":"Food", "NC":"Non-Consumable", "DR": "Drink" })
#From the Outlet_Establishment_Year field, we can find the number of years of operation of the outlet
#since this is more meaningful
data["Outlet_years"] = 2013 - data["Outlet_Establishment_Year"]
#In Item_Fat_Content, replace "Low Fat" by LF, "reg" by REG, "Regular" by REG to maintain consistency
data["Item_Fat_Content"] = data["Item_Fat_Content"].replace({"Low Fat":"LF", "low fat":"LF", "reg":"REG", "Regular": "REG" })
#If Item_Categoric_Type is Non-Consumable, the Item_Fat_Content cannot be LF. Hence changing these values to NE:Non Edible
data["Item_Fat_Content"][data.Item_Categoric_Type == "Non-Consumable"] = "NE"
#Convert Item_MRP to categorical
data["Item_MRP"] = data["Item_MRP"].tolist()
#labels = ["Low MRP", "Medium MRP", "High MRP"]
data['MRP_group'] = pd.qcut(data.Item_MRP, 3, labels=["Low MRP", "Medium MRP", "High MRP"])
#One-Hot encoding to convert categorical data to numerical values
data_mod = data
#Remove columns that have been modified
data_mod.drop(["Item_Type", "Outlet_Establishment_Year", "Item_MRP"], axis =1, inplace=True )
#data_mod = pd.concat([pd.get_dummies(data_mod["Item_Fat_Content", "Item_Categoric_Type", "Outlet_Location_Type", "Outlet_Size", "Outlet_Type" ])],  
non_dummy_cols = ["Item_Weight", "Item_Visibility", "Item_Outlet_Sales", "source", "Outlet_years", "Item_Identifier","Outlet_Identifier"]
dummy_cols = list(set(data_mod.columns) - set(non_dummy_cols))
data_mod = pd.get_dummies(data_mod, columns=dummy_cols)

#STEP5: EXPORT DATA INTO TRAIN AND TEST SETS
train_modified = data_mod.loc[data_mod["source"] == "train"]
test_modified = data_mod.loc[data_mod["source"] == "test"]
#Drop source columns from the train and test set
train_modified.drop(["source"], axis=1, inplace=True)
#train_modified = train_modified.astype(float)#see here
test_modified.drop(["source", "Item_Outlet_Sales"], axis=1, inplace=True)
#test_modified = test_modified.astype(float)
#Export the modified Train and Test sets as csv
train_modified.to_csv("C:/Users/thirumav/Desktop/Python_exercises/Big Mart/train_modified.csv", index=False)
test_modified.to_csv("C:/Users/thirumav/Desktop/Python_exercises/Big Mart/test_modified.csv", index=False)

#STEP6: MODEL BUILDING
from sklearn import cross_validation 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#Baseline model
mean_value = train_modified["Item_Outlet_Sales"].mean()   
output_item_outlet_sales = test_modified[["Item_Identifier", "Outlet_Identifier"]]
output_item_outlet_sales["Item_Outlet_Sales"] = mean_value
output_item_outlet_sales.to_csv("C:/Users/thirumav/Desktop/Python_exercises/Big Mart/Baseline_model_BigMartSales.csv", index=False)

#Defining variables used by all models
target_field = "Item_Outlet_Sales"
ID_fields = ["Item_Identifier", "Outlet_Identifier"]
feature_fields = [x for x in train_modified.columns if x not in [target_field]+ ID_fields]
cols_to_output = ["Item_Identifier", "Outlet_Identifier", "Item_Outlet_Sales"]
X_train = train_modified[feature_fields]
X_train_scaled = scaler.fit_transform(X_train)
y_train = train_modified[target_field]
X_test = test_modified[feature_fields]
X_test_scaled = scaler.transform(X_test)

###Linear Regression Model###
from sklearn.linear_model import LinearRegression
#Fitting the training data
linreg = LinearRegression(normalize = True)
linreg_fit = linreg.fit(X_train_scaled, y_train)
#Prediction on training set
y_train_prediction = linreg.predict(X_train_scaled)
#Cross-Validation
cvscore_linear = cross_validation.cross_val_score(linreg, X_train_scaled, y_train, cv = 20)
print("Cross Validation Score(20 folds): " , cvscore_linear)
print("Mean CV score for Linear Regression Model(20 folds) :{:.3f}". format(np.mean(cvscore_linear)))
print("Accuracy: %0.2f (+/- %0.2f)" % (cvscore_linear.mean(), cvscore_linear.std() * 2))
#Prediction on training set
y_train_prediction = linreg.predict(X_train_scaled)
#Prediction on test data
test_modified[target_field] = linreg.predict(X_test_scaled)
y_test = test_modified[target_field]
#Model Coefficients
coef = pd.Series(linreg.coef_, feature_fields).sort_values()
coef.plot(kind='bar', title='Model Coefficients of Linear Regression')
#Training and Test R2 score
train_score = linreg.score(X_train_scaled, y_train)
test_score = linreg.score(X_test_scaled, y_test)
print("Training data R-squared score:{:.3f} for Linear Regression".format(train_score))
print("Testing data R-squared score:{:.3f} for Linear Regression".format(test_score))
#RMSE for training data
print("RMSE score for Linear regression model:{:.3f}".format(sqrt(mean_squared_error(y_train, y_train_prediction))))
#Output CSV file
y_test_df = y_test.to_frame()
y_test_df["Item_Identifier"] = test_modified["Item_Identifier"]
y_test_df["Outlet_Identifier"] = test_modified["Outlet_Identifier"]
y_test_df = y_test_df[["Item_Identifier", "Outlet_Identifier", "Item_Outlet_Sales"]]
y_test_df.to_csv("C:/Users/thirumav/Desktop/Python_exercises/Big Mart/Linear_Regression_BigMartSales.csv", index=False)

###Ridge Regression###
from sklearn.linear_model import Ridge
for this_alpha in [0,0.05,1,10,20,50,100]:
    linridge = Ridge(alpha = this_alpha)
    #Fitting the training set
    linridge_fit = linridge.fit(X_train_scaled, y_train)
    #Cross-Validation
    cvscore_ridge = cross_validation.cross_val_score(linridge, X_train_scaled, y_train, cv = 30)
print("Accuracy: %0.2f (+/- %0.2f)" % (cvscore_ridge.mean(), cvscore_ridge.std() * 2))
#Prediction on training set
y_train_prediction = linridge.predict(X_train_scaled)
#Prediction on test data
test_modified[target_field] = linridge.predict(X_test_scaled)
y_test = test_modified[target_field]
#Model Coefficients
coef = pd.Series(linridge.coef_, feature_fields).sort_values()
coef.plot(kind='bar', title='Model Coefficients of Ridge Regression')
#Training and Test R2 score
train_score = linridge.score(X_train_scaled, y_train)
test_score = linridge.score(X_test_scaled, y_test)
print("Training data R-squared score:{:.3f} for Ridge Regression".format(train_score))
print("Testing data R-squared score:{:.3f} for Ridge Regression".format(test_score))
#RMSE for training data
print("RMSE score for Ridge regression model:{:.3f}".format(sqrt(mean_squared_error(y_train, y_train_prediction))))
#Output CSV file
y_test_df = y_test.to_frame()
y_test_df["Item_Identifier"] = test_modified["Item_Identifier"]
y_test_df["Outlet_Identifier"] = test_modified["Outlet_Identifier"]
y_test_df = y_test_df[["Item_Identifier", "Outlet_Identifier", "Item_Outlet_Sales"]]
y_test_df.to_csv("C:/Users/thirumav/Desktop/Python_exercises/Big Mart/Ridge_Regression_BigMartSales.csv", index=False)

###Lasso Regression###
from sklearn.linear_model import Lasso
linLasso = Lasso(alpha = 2.0, max_iter = 100)
linLasso_fit = linLasso.fit(X_train_scaled, y_train)
cvscore_lasso = cross_validation.cross_val_score(linLasso, X_train_scaled, y_train, cv = 30)
print("Accuracy: %0.2f (+/- %0.2f)" % (cvscore_lasso.mean(), cvscore_lasso.std() * 2))
#Prediction on training set
y_train_prediction = linLasso.predict(X_train_scaled)
#Prediction on test data
test_modified[target_field] = linLasso.predict(X_test_scaled)
y_test = test_modified[target_field]
#Model Coefficients
coef = pd.Series(linLasso.coef_, feature_fields).sort_values()
coef.plot(kind='bar', title='Model Coefficients of Lasso Regression')
#Training and Test R2 score
train_score = linLasso.score(X_train_scaled, y_train)
test_score = linLasso.score(X_test_scaled, y_test)
print("Training data R-squared score:{:.3f} for Lasso Regression".format(train_score))
print("Testing data R-squared score:{:.3f} for Lasso Regression".format(test_score))
#RMSE for training data
print("RMSE score for Lasso regression model:{:.3f}".format(sqrt(mean_squared_error(y_train, y_train_prediction))))
#Output CSV file
y_test_df = y_test.to_frame()
y_test_df["Item_Identifier"] = test_modified["Item_Identifier"]
y_test_df["Outlet_Identifier"] = test_modified["Outlet_Identifier"]
y_test_df = y_test_df[["Item_Identifier", "Outlet_Identifier", "Item_Outlet_Sales"]]
y_test_df.to_csv("C:/Users/thirumav/Desktop/Python_exercises/Big Mart/Lasso_Regression_BigMartSales.csv", index=False)

###Decision Tree###
from sklearn.tree import DecisionTreeRegressor
decisiontree = DecisionTreeRegressor(max_depth = 30, min_samples_leaf = 100, max_leaf_nodes = 50)
decisiontree.fit = decisiontree.fit(X_train_scaled, y_train)
cvscore_decision = cross_validation.cross_val_score(decisiontree, X_train_scaled, y_train, cv= 20)
print("Accuracy: %0.2f (+/- %0.2f)" % (cvscore_decision.mean(), cvscore_decision.std() * 2))
#Prediction on training set
y_train_prediction = decisiontree.predict(X_train_scaled)
#Prediction on test data
test_modified[target_field] = decisiontree.predict(X_test_scaled)
y_test = test_modified[target_field]
#Model Coefficients
coef = pd.Series(decisiontree.feature_importances_, feature_fields).sort_values()
coef.plot(kind='bar', title='Feature importances of Features in Decision tree')
#Training and Test R2 score
train_score = decisiontree.score(X_train_scaled, y_train)
test_score = decisiontree.score(X_test_scaled, y_test)
print("Training data R-squared score:{:.3f} for Decision Tree Regression".format(train_score))
print("Testing data R-squared score:{:.3f} for Decision Tree Regression".format(test_score))
#RMSE for training data
print("RMSE score for Decision Tree model:{:.3f}".format(sqrt(mean_squared_error(y_train, y_train_prediction))))
#Output CSV file
y_test_df = y_test.to_frame()
y_test_df["Item_Identifier"] = test_modified["Item_Identifier"]
y_test_df["Outlet_Identifier"] = test_modified["Outlet_Identifier"]
y_test_df = y_test_df[["Item_Identifier", "Outlet_Identifier", "Item_Outlet_Sales"]]
y_test_df.to_csv("C:/Users/thirumav/Desktop/Python_exercises/Big Mart/DecisionTree_Regression_BigMartSales.csv", index=False)

###Random Forests###
from sklearn.ensemble import RandomForestRegressor
randomforest = RandomForestRegressor(n_estimators = 10, max_depth = 6, min_samples_leaf = 100, n_jobs = 4)
randomforest.fit(X_train_scaled, y_train)
cvscore_randomforest = cross_validation.cross_val_score(randomforest, X_train_scaled, y_train, cv= 20)
print("Accuracy: %0.2f (+/- %0.2f)" % (cvscore_randomforest.mean(), cvscore_randomforest.std() * 2))
#Prediction on training set
y_train_prediction = randomforest.predict(X_train_scaled)
#Prediction on test data
test_modified[target_field] = randomforest.predict(X_test_scaled)
y_test = test_modified[target_field]
#Model Coefficients
coef = pd.Series(randomforest.feature_importances_, feature_fields).sort_values()
coef.plot(kind='bar', title='Feature importances of Features in Random Forest')
#Training and Test R2 score
train_score = randomforest.score(X_train_scaled, y_train)
test_score = randomforest.score(X_test_scaled, y_test)
print("Training data R-squared score:{:.3f} for Random Forest Regression".format(train_score))
print("Testing data R-squared score:{:.3f} for Random Forest Regression".format(test_score))
#RMSE for training data
print("RMSE score for Decision Tree model:{:.3f}".format(sqrt(mean_squared_error(y_train, y_train_prediction))))
#Output CSV file
y_test_df = y_test.to_frame()
y_test_df["Item_Identifier"] = test_modified["Item_Identifier"]
y_test_df["Outlet_Identifier"] = test_modified["Outlet_Identifier"]
y_test_df = y_test_df[["Item_Identifier", "Outlet_Identifier", "Item_Outlet_Sales"]]
y_test_df.to_csv("C:/Users/thirumav/Desktop/Python_exercises/Big Mart/RandomForest_Regression_BigMartSales.csv", index=False)

###Gradient Boosting Regressor###
from sklearn.ensemble import GradientBoostingRegressor
#Choosing best estimator
estimators = [50, 75, 80, 100, 125, 150, 200, 250, 500]
for e in estimators:
        gbr = GradientBoostingRegressor(random_state = 1, n_estimators = e, min_samples_split = 8, 
                                            min_samples_leaf = 4, learning_rate = 0.1)
        gbr.fit(X_train_scaled, y_train)
        cvscore_gradientboost = cross_validation.cross_val_score(gbr, X_train_scaled, y_train, cv= 5)
        print("Accuracy: %0.2f (+/- %0.2f), Estimator: %0.1f " % (cvscore_gradientboost.mean(), cvscore_gradientboost.std() * 2, e))
#Since for estimator=80, maximum cv accuracy is obtained, it is chosen for the Gradient Boosting regression model
gradientboost = GradientBoostingRegressor(random_state = 1, n_estimators = 80, min_samples_split = 8, 
                                        min_samples_leaf = 4, learning_rate = 0.1) 
gradientboost.fit(X_train_scaled, y_train)   
#Prediction on training set
y_train_prediction = gradientboost.predict(X_train_scaled)
#Prediction on test data
test_modified[target_field] = gradientboost.predict(X_test_scaled)
y_test = test_modified[target_field]
#Model Coefficients
coef = pd.Series(gradientboost.feature_importances_, feature_fields).sort_values()
coef.plot(kind='bar', title='Feature importances of Features in Gradient Boost')
#Training and Test R2 score
train_score = gradientboost.score(X_train_scaled, y_train)
test_score = gradientboost.score(X_test_scaled, y_test)
print("Training data R-squared score:{:.3f} for Gradient Boost Regression".format(train_score))
print("Testing data R-squared score:{:.3f} for Gradient Boost Regression".format(test_score))
#RMSE for training data
print("RMSE score for Gradient Boosting Regressor model:{:.3f}".format(sqrt(mean_squared_error(y_train, y_train_prediction))))
#Output CSV file
y_test_df = y_test.to_frame()
y_test_df["Item_Identifier"] = test_modified["Item_Identifier"]
y_test_df["Outlet_Identifier"] = test_modified["Outlet_Identifier"]
y_test_df = y_test_df[["Item_Identifier", "Outlet_Identifier", "Item_Outlet_Sales"]]
y_test_df.to_csv("C:/Users/thirumav/Desktop/Python_exercises/Big Mart/GradientBoost_Regression_BigMartSales.csv", index=False)
    
###MLP Regressor###
from sklearn.neural_network import MLPRegressor
for thisactivation in ('tanh','relu'):
    for thisalpha in (0.0001, 1.0, 100):
        mlpReg = MLPRegressor(activation = thisactivation, random_state =0, 
                              alpha = thisalpha, solver = 'lbfgs')
        mlpReg.fit(X_train_scaled, y_train)
        cvscore_mlpRegressor = cross_validation.cross_val_score(mlpReg, X_train_scaled, y_train, cv= 20)
        print("Accuracy: %0.2f (+/- %0.2f), Alpha: %0.4f" % (cvscore_mlpRegressor.mean(), cvscore_mlpRegressor.std() * 2, thisalpha))

mlpRegressor = MLPRegressor(activation = 'relu', random_state =0, 
                              alpha = 100, solver = 'lbfgs')
mlpRegressor.fit(X_train_scaled, y_train)
#Prediction on training set
y_train_prediction = mlpRegressor.predict(X_train_scaled)
#Prediction on test data
test_modified[target_field] = mlpRegressor.predict(X_test_scaled)
y_test = test_modified[target_field]
#Training and Test R2 score
train_score = mlpReg.score(X_train_scaled, y_train)
test_score = mlpReg.score(X_test_scaled, y_test)
print("Training data R-squared score:{:.3f} for MLP Regression".format(train_score))
print("Testing data R-squared score:{:.3f} for MLP Regression".format(test_score))
#RMSE for training data
print("RMSE score for MLP Regressor model:{:.3f}".format(sqrt(mean_squared_error(y_train, y_train_prediction))))
#Output CSV file
y_test_df = y_test.to_frame()
y_test_df["Item_Identifier"] = test_modified["Item_Identifier"]
y_test_df["Outlet_Identifier"] = test_modified["Outlet_Identifier"]
y_test_df = y_test_df[["Item_Identifier", "Outlet_Identifier", "Item_Outlet_Sales"]]
y_test_df.to_csv("C:/Users/thirumav/Desktop/Python_exercises/Big Mart/MLP_Regression_BigMartSales.csv", index=False)

        