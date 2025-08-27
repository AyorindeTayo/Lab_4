# Lab: Data Preprocessing Techniques (Based on Chapter 4 of Python Machine Learning, 2nd Edition)

## Objectives
By the end of this lab, you will be able to:
- Identify and handle missing values in datasets using pandas and scikit-learn.
- Encode categorical features, distinguishing between ordinal and nominal types.
- Partition datasets into training and test sets.
- Scale features using normalization and standardization.
- Perform feature selection using regularization, sequential backward selection, and random forest importance.

## Requirements
- Python 3.x
- Libraries: pandas, numpy, scikit-learn, matplotlib
- Dataset: Wine dataset (load from URL or local file)

Install required libraries if needed:
```
pip install pandas numpy scikit-learn matplotlib
```

## Section 1: Handling Missing Data

### Background
Missing data can arise from errors or omissions. We need to handle it by removing or imputing values to avoid issues in machine learning algorithms.

### Exercise 1.1: Identifying Missing Values
Create a sample DataFrame with missing values and identify them.

```python
import pandas as pd
from io import StringIO

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''
df = pd.read_csv(StringIO(csv_data))
print(df)
print(df.isnull().sum())
```

**Task:** 
- Run the code. How many missing values are in each column?
- Modify the DataFrame to add more missing values and recheck.

### Exercise 1.2: Eliminating Missing Values
Remove rows or columns with missing values.

```python
# Drop rows with any missing values
print(df.dropna(axis=0))

# Drop columns with any missing values
print(df.dropna(axis=1))

# Drop rows where all values are missing (none in this case)
print(df.dropna(how='all'))

# Drop rows with fewer than 4 non-missing values
print(df.dropna(thresh=4))

# Drop rows where 'C' is missing
print(df.dropna(subset=['C']))
```

**Task:** 
- Experiment with different parameters. When would dropping rows vs. columns be preferable?

### Exercise 1.3: Imputing Missing Values
Use scikit-learn's SimpleImputer (note: Imputer is deprecated; use SimpleImputer in newer versions).

```python
from sklearn.impute import SimpleImputer
import numpy as np

imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
print(imputed_data)
```

**Task:** 
- Change strategy to 'median' or 'most_frequent'. Compare results.
- Why might 'most_frequent' be useful for categorical data?

## Section 2: Handling Categorical Data

### Background
Categorical data needs encoding: ordinal (ordered) via mapping, nominal (unordered) via one-hot encoding. Class labels should be integers.

### Exercise 2.1: Mapping Ordinal Features
```python
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']])
df.columns = ['color', 'size', 'price', 'classlabel']

size_mapping = {'XL': 3, 'L': 2, 'M': 1}
df['size'] = df['size'].map(size_mapping)
print(df)

# Inverse mapping
inv_size_mapping = {v: k for k, v in size_mapping.items()}
print(df['size'].map(inv_size_mapping))
```

**Task:** 
- Add more sizes (e.g., 'S': 0). Update the DataFrame and mapping.

### Exercise 2.2: Encoding Class Labels
```python
from sklearn.preprocessing import LabelEncoder

class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y)

# Inverse
print(class_le.inverse_transform(y))
```

**Task:** 
- Why encode labels as integers? Test with a classifier if not encoded.

### Exercise 2.3: One-Hot Encoding for Nominal Features
```python
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print(X)

# One-hot encoding
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
print(ohe.fit_transform(X[:, [0]]).toarray())  # Only on 'color'

# Using pandas get_dummies
print(pd.get_dummies(df[['price', 'color', 'size']]))

# Drop first column to avoid multicollinearity
print(pd.get_dummies(df[['price', 'color', 'size']], drop_first=True))
```

**Task:** 
- Apply one-hot encoding to the full DataFrame. Discuss multicollinearity and why dropping one column helps.

## Section 3: Partitioning a Dataset

### Background
Split data into train/test sets for unbiased evaluation.

### Exercise 3.1: Loading and Splitting the Wine Dataset
```python
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
print(df_wine.head())

from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
print('Train shape:', X_train.shape, 'Test shape:', X_test.shape)
```

**Task:** 
- Change test_size to 0.2. Check class proportions in y_train and y_test.
- Why use stratify=y?

## Section 4: Feature Scaling

### Background
Scale features for better performance in distance-based or gradient algorithms.

### Exercise 4.1: Normalization (Min-Max Scaling)
```python
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)
print(X_train_norm[:2])  # First two rows
```

**Task:** 
- Plot histograms of a feature before/after scaling.

### Exercise 4.2: Standardization
```python
from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
print(X_train_std[:2])
```

**Task:** 
- Compare mean and std of scaled features (should be ~0 and ~1).
- When is standardization preferred over normalization?

## Section 5: Selecting Meaningful Features

### Background
Reduce overfitting via regularization or feature selection.

### Exercise 5.1: L1 Regularization for Sparsity
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', multi_class='ovr')
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))
print(lr.coef_)
```

**Task:** 
- Vary C (e.g., 0.1, 10). Observe sparsity in coefficients.
- Plot coefficients vs. C as in the chapter.

### Exercise 5.2: Sequential Backward Selection (SBS)
Implement SBS as in the chapter (copy the class code).

```python
# Paste SBS class here

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

knn = KNeighborsClassifier(n_neighbors=5)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

# Selected features
k3 = list(sbs.subsets_[10])
print(df_wine.columns[1:][k3])

# Performance with selected
knn.fit(X_train_std[:, k3], y_train)
print('Test accuracy:', knn.score(X_test_std[:, k3], y_test))
```

**Task:** 
- Change k_features or estimator. Analyze the plot.

### Exercise 5.3: Feature Importance with Random Forests
```python
from sklearn.ensemble import RandomForestClassifier

feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()
```

**Task:** 
- Use SelectFromModel with threshold=0.1. Train a model on selected features.
- Compare top features from RF and SBS.

## Conclusion
Summarize what you learned. Experiment with a different dataset (e.g., Iris) and apply these techniques. Discuss how preprocessing affects model performance.

**Questions:**
1. Why is handling missing data important?
2. When should you use one-hot encoding vs. label encoding?
3. How does feature scaling impact algorithms like KNN or SVM?
4. Compare L1 regularization and SBS for feature selection.
