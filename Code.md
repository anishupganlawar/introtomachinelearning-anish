# Code & Explanation

# Basic Data Exploration

---

### **1. Load the Dataset**

```python
import pandas as pd

data = pd.read_csv("path/to/file.csv")
```

**Explanation:**  
- Imports pandas  
- Reads the CSV file and loads it into a DataFrame called `data`  

---

### **2. View the First Few Rows**

```python
data.head()
```

**Explanation:**  
Displays the first 5 rows of the dataset. Helps you understand what the data looks like.

---

### **3. Summary Statistics**

```python
data.describe()
```

**Explanation:**  
Shows count, mean, std, min, max, and percentiles for numeric columns.

---

### **4. Select a Single Column**

```python
data["SalePrice"]
```

**Explanation:**  
Extracts one column from the dataset as a pandas Series.

---

### **5. Select Multiple Columns**

```python
columns = ["LotArea", "YearBuilt", "1stFlrSF", "FullBath"]
subset = data[columns]
```

**Explanation:**  
Creates a new DataFrame containing only the selected columns.

---

### **6. Count Missing Values**

```python
data.isnull().sum()
```

**Explanation:**  
Counts

## **7. Check Data Types**

```python
data.dtypes
```

**Explanation:**  
Shows whether each column is int, float, object (text), etc.

---

## **8. Explore Unique Values (for categorical/text columns)**

```python
data["Neighborhood"].unique()
```

**Explanation:**  
Lists all unique categories in the selected column.

---

## **9. Basic Visual Exploration**

### Histogram

```python
data["SalePrice"].hist()
```

**Explanation:**  
Shows the distribution of a numeric feature.

---

### Scatterplot

```python
data.plot.scatter("GrLivArea", "SalePrice")
```

**Explanation:**  
Helps visualize the relationship between two numeric variables.

---
<br>

# Your First Machine Learning Model 

---

### **1. Load the Dataset**

```python
import pandas as pd

data = pd.read_csv("path/to/file.csv")
```

**Explanation:**  
Loads the dataset into a DataFrame so we can explore and model it.

---

### **2. Select the Target Column (y)**

```python
y = data["SalePrice"]
```

**Explanation:**  
`y` is what we want to predict.  
Here, we are predicting the house’s sale price.

---

### **3. Select Feature Columns (X)**

```python
feature_names = ["LotArea", "BedroomAbvGr", "FullBath", "TotRmsAbvGrd"]
X = data[feature_names]
```

**Explanation:**  
Creates a DataFrame with only the chosen input columns (features).  
These are the "clues" the model uses to make predictions.

---

### **4. Build the Decision Tree Model**

```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state=1)
```

**Explanation:**  
Creates a Decision Tree model.  
`random_state=1` ensures the results are the same every time you run it.

---

### **5. Fit the Model (Training)**

```python
model.fit(X, y)
```

**Explanation:**  
The model “learns” patterns in the data.  
After this step, the model can make predictions.

---

### **6. Make Predictions**

```python
predictions = model.predict(X)
```

**Explanation:**  
Uses the trained model to predict sale prices using the same training data.

---

### **7. Look at the First Few Predictions**

```python
predictions[:5]
```

**Explanation:**  
Shows the first 5 predicted prices.  
Useful for checking that the model actually output numbers.

---

### **8. Compare Predictions to Actual Values**

```python
comparison = pd.DataFrame({"Actual": y[:5], "Predicted": predictions[:5]})
comparison
```

**Explanation:**  
Helps visualize how close (or far) the predictions are from the real prices.

---
<br> 

#  Model Validation 

This section shows how to split data into training and validation sets, train a model, and evaluate it using MAE (Mean Absolute Error).

---

### **1. Load the Data**

```python
import pandas as pd

data = pd.read_csv("path/to/file.csv")
```

**Explanation:**  
Reads the dataset so we can prepare features and target values.

---

### **2. Select Target (y) and Features (X)**

```python
y = data["SalePrice"]

features = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF",
            "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]

X = data[features]
```

**Explanation:**  
- `y` is the value we want the model to predict  
- `X` contains the columns used to make predictions  

---

### **3. Split Data into Training + Validation Sets**

```python
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(
    X, y, random_state=1
)
```

**Explanation:**  
Splits the data into two parts:  
- **Training set** → used to fit the model  
- **Validation set** → used to test the model on unseen data  

`random_state=1` ensures reproducible results.

---

### **4. Build the Model**

```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state=1)
```

**Explanation:**  
Creates a decision tree model that will learn patterns from the training data.

---

### **5. Train the Model**

```python
model.fit(train_X, train_y)
```

**Explanation:**  
The model learns the relationship between the features and the target.

---

### **6. Make Predictions on the Validation Set**

```python
val_predictions = model.predict(val_X)
```

**Explanation:**  
Uses the trained model to make predictions on data the model has *never seen before*.

---

### **7. Evaluate Using Mean Absolute Error**

```python
from sklearn.metrics import mean_absolute_error

val_mae = mean_absolute_error(val_y, val_predictions)
print(val_mae)
```

**Explanation:**  
MAE measures how far off the model’s predictions are, on average.  
Lower MAE = better model.

---
<br>

# Underfitting & Overfitting 

This section shows how changing model complexity (max_leaf_nodes) affects performance. We evaluate each version using MAE. <br>
<br> **Mean Absolute Error (MAE)** <br>
MAE tells on average, how far off are predictions from the real values?
It looks at every prediction the model made, compares it to the actual value, measures the difference, and then averages all those differences.
<br> <br>
MAE = average(|actual – predicted|)
<br> <br>
If MAE = 20,000 it literally means: “Your predicted house prices are wrong by $20,000 on average.”

---

### **1. Load the Data**

```python
import pandas as pd

data = pd.read_csv("path/to/file.csv")
```

**Explanation:**  
Reads the dataset into a DataFrame.

---

### **2. Select Target (y) and Features (X)**

```python
y = data["SalePrice"]

features = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF",
            "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]

X = data[features]
```

**Explanation:**  
- `y` → value to predict  
- `X` → columns used to make predictions  

---

### **3. Split into Training + Validation**

```python
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(
    X, y, random_state=1
)
```

**Explanation:**  
We use the training set to fit the model and the validation set to test how well it generalizes.

---

### **4. Define a Function to Evaluate MAE for Given Tree Size**

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    predictions = model.predict(val_X)
    mae = mean_absolute_error(val_y, predictions)
    return mae
```

**Explanation:**  
This function:  
- builds a tree with a given number of leaf nodes  
- trains it  
- predicts on validation data  
- returns the MAE  

Used to compare model complexity levels.

---

### **5. Compare Models with Different Tree Sizes**

```python
candidate_nodes = [5, 25, 50, 100, 250, 500]

for max_leaf_nodes in candidate_nodes:
    mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print(max_leaf_nodes, mae)
```

**Explanation:**  
Tests multiple tree sizes to see which produces the lowest MAE.  
- **Too small tree → underfitting**  
- **Too large tree → overfitting**  
- **Best tree → lowest MAE**

---

### **6. Choose the Best Tree Size (Example)**

```python
best_tree_size = 100  # based on printed MAE values
```

**Explanation:**  
Whichever tree size gives the lowest MAE is selected as the best model.

---
<br>

# Random Forests 

This section shows how to train a Random Forest model and compare it to a Decision Tree using MAE.

---

### 1. Load the Data

```python
import pandas as pd

data = pd.read_csv("path/to/file.csv")
```

**Explanation:**  
Loads the housing dataset so we can train models.

---

### 2. Select Target (y) and Features (X)

```python
y = data["SalePrice"]

features = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF",
            "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]

X = data[features]
```

**Explanation:**  
- `y` = the value we want to predict  
- `X` = columns used for prediction  

---

### 3. Split Into Training + Validation Sets

```python
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(
    X, y, random_state=1
)
```

**Explanation:**  
Ensures the model is evaluated on data it hasn't seen before.

---

### 4. Train a Decision Tree (Benchmark Model)

```python
from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor(random_state=1)
dt_model.fit(train_X, train_y)
```

**Explanation:**  
A simple baseline model to compare against the Random Forest.

---

### 5. Evaluate the Decision Tree Using MAE

```python
from sklearn.metrics import mean_absolute_error

dt_preds = dt_model.predict(val_X)
dt_mae = mean_absolute_error(val_y, dt_preds)

print("Decision Tree MAE:", dt_mae)
```

**Explanation:**  
Shows how well (or poorly) a single tree performs.

---

### 6. Train a Random Forest Model

```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
```

**Explanation:**  
Random Forest builds MANY trees and averages their predictions.  
This usually improves accuracy.

---

### 7. Evaluate the Random Forest Using MAE

```python
rf_preds = rf_model.predict(val_X)
rf_mae = mean_absolute_error(val_y, rf_preds)

print("Random Forest MAE:", rf_mae)
```

**Explanation:**  
Random Forest almost always produces a *lower* MAE compared to a single tree.

---

### 8. Compare the Two Models

```python
print("Decision Tree MAE:", dt_mae)
print("Random Forest MAE:", rf_mae)
```

**Explanation:**  
Shows which model performs better.  
The Random Forest should have a lower MAE.

---





