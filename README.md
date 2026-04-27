# 🚀 AI & Machine Learning Bootcamp Repository

> A structured, implementation-driven guide to Machine Learning — from intuition to production-ready systems.

---

# 🧭 Table of Contents

* [Introduction to AI](#-introduction-to-artificial-intelligence)
* [Introduction to ML](#-introduction-to-machine-learning)
* [Types of ML](#-types-of-machine-learning)
* [Problem Formulation](#-problem-formulation)
* [ML Pipeline](#-machine-learning-pipeline-deep-dive)
* [Libraries & Tools](#-libraries--tools)
* [EDA](#-dataset-understanding-eda)
* [Implementation](#-step-by-step-implementation)
* [Pipelines](#-scikit-learn-pipeline)
* [Resources](#-resources)

---

# 🧠 Introduction to Artificial Intelligence

Artificial Intelligence focuses on building systems that can:

* Learn from data
* Make decisions
* Adapt over time

> AI = Intelligent Agents (Perception → Decision → Action)

---

# 🤖 Introduction to Machine Learning

Machine Learning enables systems to **learn patterns from data** instead of hardcoded rules.

### 🧩 ML as Function Learning

```
f(X) → Y
```

* X → Features
* Y → Target

---

# 📊 Types of Machine Learning

### 🟦 Supervised Learning

* Regression
* Classification

### 🟩 Unsupervised Learning

* Clustering
* Dimensionality Reduction

---

# 🎯 Problem Formulation

A correct ML problem must define:

* Input Features (X)
* Output Target (Y)
* Objective Function

⚠️ Bad problem formulation = useless model

---

# 🔄 Machine Learning Pipeline (Deep Dive)

---

## 🟦 Step 1: Data Extraction

Sources:

* CSV / Excel
* Databases
* APIs

---

## 🟩 Step 2: Data Ingestion (Improved)

```python
import pandas as pd

# Advanced data loading with real-world considerations
df = pd.read_csv(
    "data/dataset.csv",
    sep=",",
    encoding="utf-8",
    na_values=["NA", "?", "null"],
    parse_dates=True,
    infer_datetime_format=True,
    low_memory=False
)

# Initial inspection
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nPreview:\n", df.head())

# Info & stats
print("\nInfo:\n")
df.info()

print("\nSummary Stats:\n")
print(df.describe(include="all"))
```

📎 **Footnote — pandas.read_csv()**

* Loads CSV into DataFrame
* Supports missing values handling (`na_values`)
* Optimizes memory usage (`low_memory`)
* Can parse datetime automatically

---

## 🟨 Step 3: Data Understanding

```python
# Missing values
missing = df.isnull().sum().sort_values(ascending=False)
print("Missing Values:\n", missing.head(10))

# Duplicates
print("Duplicate Rows:", df.duplicated().sum())

# Unique values
print("Unique Values:\n", df.nunique())
```

📎 **Footnote — DataFrame methods**

* `isnull()` → Detect missing data
* `duplicated()` → Detect duplicate rows
* `nunique()` → Count unique values

---

## 🟥 Step 4: Data Cleaning (Realistic)

```python
# Drop duplicates
df = df.drop_duplicates()

# Fill missing values intelligently
for col in df.select_dtypes(include="number").columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna(df[col].mode()[0])
```

📎 **Footnote — Strategy**

* Numerical → median (robust to outliers)
* Categorical → mode

---

## 🔵 Step 5: Feature Engineering

```python
# Example: Encoding categorical variables
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]
```

📎 **Footnote — get_dummies()**
Converts categorical variables into numerical format using one-hot encoding.

---

## 🧠 Production Insight

In real-world ML systems:

* Data pipelines use ETL workflows
* Data is versioned (DVC)
* Validation pipelines ensure quality

---

# 🧰 Libraries & Tools

* NumPy → Numerical computing
* pandas → Data manipulation
* scikit-learn → ML models
* Matplotlib → Visualization

---

# 📊 Dataset Understanding (EDA)

```python
import matplotlib.pyplot as plt

# Distribution plots
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()
```

📎 **Footnote — hist()**
Quick way to visualize feature distributions.

---

# ⚙️ Step-by-Step Implementation

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
```

📎 **Footnote — LogisticRegression**

* Linear model for classification
* Works well for baseline models

---

# 🔗 Scikit-learn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
```

📎 **Footnote — Pipeline**

* Chains preprocessing + model
* Prevents data leakage
* Cleaner code

---

# 📚 Resources

* Official Documentation:

  * pandas
  * scikit-learn

---

# ⭐ Final Note

This repository is designed to:

* Build strong ML fundamentals
* Provide practical implementation
* Prepare for real-world ML systems

---

# 🚀 Future Scope

* Deep Learning (CNN, RNN, LSTM)
* Deployment (FastAPI, Docker)
* MLOps (CI/CD, Monitoring)

---
