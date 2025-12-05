# Introduction to Machine Learning 
## How Models Work?
- If the model sees certain things in the data, then it makes a certain prediction.

Why we need a model?
- Imagine your cousin has made money predicting house prices.
He says it’s “intuition,” but really, he’s just seen enough houses to notice patterns.
Machine learning is the same thing, except the computer learns the patterns for us. we have to be careful though: if the model learns the wrong patterns or too few patterns, then it makes bad predictions.

Example of a model: "a decision tree", it works like a small checklist <br> 
- If a house has more than 2 bedrooms, → then the model predicts a higher price. <br>
- If it has 2 or fewer bedrooms,→ then the model predicts a lower price. <br>
That’s basically the entire mindset of a decision tree: <br>
- ask a question → split the data → make a guess. <br>

Machine learning models make predictions by spotting patterns.
If they learn the right patterns, they work well.
If they don’t, everything breaks.
<br>
<br>

## Basic Data Exploration
- before building any machine learning model, the first thing we have to do is get familiar with the data.

Using Pandas to Explore the Data <br>
- we start by loading the dataset using Pandas: **import pandas as pd**

Pandas gives us a structure called a DataFrame, which works like an Excel sheet inside Python. Once the data is loaded, we can use simple commands to inspect it. <br>
- One of the most helpful tools is: **data.describe()** <br> <br>
This gives us a quick summary of every numeric column in the dataset. <br>

**Interpreting the Summary**  
The `.describe()` output shows 8 key numbers for each column:

- **count** → how many non-missing values exist  
- **mean** → the average  
- **std** → how spread out the values are  
- **min** → the smallest value  
- **25%** → 25% of the values are smaller than this (25th percentile)  
- **50% (median)** → half the values are below this  
- **75%** → 75% of values are below this  
- **max** → the largest value  

This helps us understand the shape of the data.  

For example:  
If the minimum number of bedrooms is 1 and the maximum is 10, we know our dataset includes everything from tiny apartments to very large homes.

We also learn about missing values, if **count** is less than the total number of rows, then some data points are missing. That’s something we need to be careful about later.


## Our First Machine Learning Model
Before building a model, we first choose which columns of our dataset to use. The dataset has many variables, and not all of them are useful. So we start by checking the list of columns and picking the ones that make sense to us.<br><br>

We also need to be careful about missing values. If some columns have missing data, we either drop those rows or handle the missing values later. For now, we take the simple option: dropping the rows with missing values.<br><br>

Next, we select the **prediction target**, which is the column we want the model to predict. In this example, the target is the house price.<br><br>

Then we choose our **features**, which are the columns the model will use to make predictions. We pick a few features like Rooms, Bathroom, Landsize, Latitude, and Longitude. These become the input to the model.<br><br>

Once we have our features (X) and our target (y), we build the model using a **DecisionTreeRegressor** from scikit-learn. Every machine learning workflow follows the same steps:<br>
1. **Define** → choose the type of model<br>
2. **Fit** → let the model learn patterns from the data<br>
3. **Predict** → use the model to guess new values<br>
4. **Evaluate** → check how accurate the predictions are<br><br>

After fitting the model with X and y, we can make predictions. we predict prices for the first few houses in the dataset, and the model returns estimated prices based on the features we selected.<br><br>

This is the basic pattern behind every machine learning model. <br>


## Model Validation
When we build a model, we want to know one thing:<br>
**“Is this thing actually good, or is it just pretending?”**<br><br>

A lot of people check accuracy by testing the model on the **same data** it learned from.<br>
That’s like checking if a student is smart by giving them the exact same worksheet they memorized.  
Of course they’ll score high.<br><br>

But in the real world, the model will see **new** data.<br>
If it fails there → the model is useless.<br><br>

So model validation is basically:<br>
➡️ **Train on one part of the data**<br>
➡️ **Test on a different part**<br><br>

If we get this wrong, we get fooled by “fake accuracy.”  
If we do it right, we know exactly how our model performs on fresh, unseen stuff.<br><br>

### The Big Problem With Training Accuracy<br>
If you only test on training data:<br>
- The model will look perfect.<br>
- But that performance is fake confidence.<br>
- When you give it new data, the errors explode.<br><br>

Example:<br>
- In-sample error ≈ **500 dollars**<br>
- Out-of-sample error ≈ **250,000 dollars**<br><br>

That’s not a small mistake.<br>
That’s your model saying a house costs ₹2 crore when it's ₹80 lakh.<br><br>

This is why validation matters.<br><br>

### **Why We Split the Data**<br>
We split the data into two parts using `train_test_split`:<br><br>

- **train_X, train_y** → model learns here<br>
- **val_X, val_y** → model is tested here<br><br>

This lets us see how the model performs on data it hasn’t seen before.<br><br>

If the validation error is high, it means:<br>
- The model memorized the training data<br>
- It didn’t actually learn general patterns<br>
- We need a better model or better features<br><br>

### **MAE (Mean Absolute Error) — Simple Version**<br>
MAE tells us: “On average, how wrong is the model?”<br><br>

If **MAE = 250,000**, that means:<br>
“Every prediction is off by about 250k.”<br><br>

- Big MAE → bad model<br>
- Small MAE → better model<br><br>


## Underfitting and Overfitting
When we train a model, we have to balance two things. If we make the model too simple, it won’t learn enough patterns and that’s **underfitting**. If we make it too complex, it memorizes the training data and that’s **overfitting**.<br><br>

A decision tree has “depth,” which basically decides how many times it can split the data. More depth = more splits = more patterns learned. But there’s a catch:<br>
- If the tree is **too shallow**, it misses important differences → **underfitting**.<br>
- If the tree is **too deep**, it starts memorizing random noise → **overfitting**.<br><br>

We check this using **validation data**, data the model hasn’t seen while training. If the training error is super low but the validation error suddenly jumps, the model is overfitting. If both errors are high, we’re underfitting.<br><br>

When we try different values for `max_leaf_nodes`, we see how accuracy changes:<br>
- 5 leaves → bad accuracy (too simple)<br>
- 50 leaves → better<br>
- **500 leaves → best** (sweet spot)<br>
- 5000 leaves → worse again (too complex)<br><br>

**The goal:** find the spot where the validation error is the lowest. That’s where the model learns enough to be smart, but not enough to cheat.<br><br>

**Quick takeaway:**<br>
- **Underfitting** → model is clueless<br>
- **Overfitting** → model is overconfident and wrong<br>
- **Validation data** → helps us keep the model honest<br><br>

This is the balancing act every ML model goes through, and we tune it until we hit that sweet spot where the model actually performs well on new data.

## Random Forests

Random Forests basically say:  
**“Instead of trusting one decision tree, let’s ask a bunch of trees and average their answers.”**

Here’s what that means for us:

- **If one tree overfits**, the others pull it back.  
- **If one tree underfits**, the group keeps the prediction stable.  
- **Together**, they make a more reliable model than any single tree.

So instead of relying on one model's tree, we use many trees trained on slightly different slices of the data.  
That cuts down noise and makes predictions way more consistent.

### **Why this matters for us**

- **A single deep tree → overfits**  
- **A single shallow tree → underfits**  
- **A Random Forest → balances everything automatically**

When we ran it, the MAE dropped to **191,669**, which is a huge improvement over the ~250k error from the best single decision tree.

That tells us the forest is doing exactly what we want:  
**reducing randomness, smoothing out errors, and giving us better predictions without needing much tuning.**





