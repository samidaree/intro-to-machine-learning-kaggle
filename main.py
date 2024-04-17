import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
 

# STEP 1 : Load Data and extract DataFrame
melbourne_file_path = './input/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
print(melbourne_data.describe())

# The target property we want to predict (target prediction)
y = melbourne_data.Price

#Properties (called features) we want to use for prediction of the target prediction y (price)
melbourne_features = ["Rooms","Bathroom","Landsize","Lattitude", "Longtitude"]


#DataFrame restricted to the features we want to use for prediction
X = melbourne_data[melbourne_features]
print("\nDataFrame X : ")
print(X)

# STEP 2 : Split data into training and validation data, for both features and target
print(train_test_split(X,y,random_state =1))
train_X,validation_X, train_y, validation_y = train_test_split(X,y,random_state =1)

print("\nTraining data train_X : ")
print(train_X)

print("\nValidation data validation_X : ")
print(validation_X)

print("\nTarget column property of the training data train_y : ")
print(train_y)

print("\nTarget column property of the validation data validation_y: ")
print(validation_y)



#STEP 3 : Specify and Fit the model with Random Forest Model
# Define the model. Set random_state to 1
melbourne_rfm_model = RandomForestRegressor(random_state=1)
# fit your model
melbourne_rfm_model.fit(train_X, train_y)

#STEP 4: Make Predictions with Validation Data and Calculate Mean Absolute Error 
prediction = melbourne_rfm_model.predict(validation_X)
# Calculate the mean absolute error of your Random Forest model on the validation Prediction Target
melbourne_rfm_mae = mean_absolute_error(validation_y,prediction)

print("Validation MAE for Random Forest Model: {}".format(melbourne_rfm_mae))

# STEP 3b : DecisionTree
melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(train_X,train_y)
print("\nMaking predictions for the following 5 houses : ")
print(validation_X.head())
print("\nThe predictions are ") 

# STEP 4b : 
predicted_home_prices = melbourne_model.predict(validation_X)
print(predicted_home_prices)
error = mean_absolute_error(validation_y, predicted_home_prices)
print("\nError : ", error)

#OPTIONAL : Find the best tree depth and the minimum Mean Absolute Error
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_depth = candidate_max_leaf_nodes[0]
min = get_mae(best_tree_depth,train_X,val_X,train_y,val_y)
for max_leaf_nodes in candidate_max_leaf_nodes : 
    my_mae = get_mae(max_leaf_nodes,train_X, val_X, train_y, val_y)
    if (my_mae<min) :
        best_tree_depth = max_leaf_nodes
        min = my_mae
    
print(best_tree_depth)
final_model = DecisionTreeRegressor(max_leaf_nodes = best_tree_depth, random_state=0)
# fit the final model 
final_model.fit(X, y)


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

