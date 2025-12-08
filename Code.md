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
