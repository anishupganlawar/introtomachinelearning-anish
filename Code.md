# Code & Explanation

## Basic Data Exploration

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

### 📊 Histogram

```python
data["SalePrice"].hist()
```

**Explanation:**  
Shows the distribution of a numeric feature.

---

### 📈 Scatterplot

```python
data.plot.scatter("GrLivArea", "SalePrice")
```

**Explanation:**  
Helps visualize the relationship between two numeric variables.

---

## Your First Machine Learning Model 

---

## **1. Load the Dataset**

```python
import pandas as pd

data = pd.read_csv("path/to/file.csv")
```

**Explanation:**  
Loads the dataset into a DataFrame so we can explore and model it.

---

## **2. Select the Target Column (y)**

```python
y = data["SalePrice"]
```

**Explanation:**  
`y` is what we want to predict.  
Here, we are predicting the house’s sale price.

---

## **3. Select Feature Columns (X)**

```python
feature_names = ["LotArea", "BedroomAbvGr", "FullBath", "TotRmsAbvGrd"]
X = data[feature_names]
```

**Explanation:**  
Creates a DataFrame with only the chosen input columns (features).  
These are the "clues" the model uses to make predictions.

---

## **4. Build the Decision Tree Model**

```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state=1)
```

**Explanation:**  
Creates a Decision Tree model.  
`random_state=1` ensures the results are the same every time you run it.

---

## **5. Fit the Model (Training)**

```python
model.fit(X, y)
```

**Explanation:**  
The model “learns” patterns in the data.  
After this step, the model can make predictions.

---

## **6. Make Predictions**

```python
predictions = model.predict(X)
```

**Explanation:**  
Uses the trained model to predict sale prices using the same training data.

---

## **7. Look at the First Few Predictions**

```python
predictions[:5]
```

**Explanation:**  
Shows the first 5 predicted prices.  
Useful for checking that the model actually output numbers.

---

## **8. Compare Predictions to Actual Values (Optional)**

```python
comparison = pd.DataFrame({"Actual": y[:5], "Predicted": predictions[:5]})
comparison
```

**Explanation:**  
Helps visualize how close (or far) the predictions are from the real prices.

---
