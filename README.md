# 🚀 AI & Machine Learning Bootcamp Repository
*A structured, implementation-first guide to Machine Learning — from intuition to production-ready workflows.*

---
## 🧭 Table of Contents
1. [Introduction to Artificial Intelligence](#-1-introduction-to-artificial-intelligence)  
2. [Introduction to Machine Learning](#-2-introduction-to-machine-learning)  
3. [Current State of AI](#-3-current-state-of-ai)  
4. [Types of Machine Learning](#-4-types-of-machine-learning)  
5. [Problem Formulation](#-5-problem-formulation)  
6. [Machine Learning Pipeline](#-6-machine-learning-pipeline-deep-dive)  
7. [Libraries & Tools](#-7-libraries--tools)  
8. [Dataset Understanding (EDA)](#-8-dataset-understanding-eda)  
9. [Model Implementation](#-9-model-implementation)  
10. [Scikit-learn Pipeline](#-10-scikit-learn-pipeline-advanced)  
11. [Resources](#-11-resources)  

---

## 🧠 1. Introduction to Artificial Intelligence

Artificial Intelligence (AI) refers to systems that can:

* Learn from data  
* Make decisions  
* Adapt over time  

AI systems are intelligent agents that perceive and act to achieve goals.

---

## 🤖 2. Introduction to Machine Learning

Machine Learning enables systems to learn patterns from data instead of hardcoded rules.

### 🧩 ML as Function Learning

```
f(X) → Y
```

* **X → Features**  
* **Y → Target**

---

## 🌐 3. Current State of AI

* 🤖 AI Agents  
* 🧠 Large Language Models  
* 🎨 Generative AI  

⚠️ All modern AI systems rely on ML pipelines.

---

## 📊 4. Types of Machine Learning

### 🟦 Supervised Learning
* Regression  
* Classification  

### 🟩 Unsupervised Learning
* Clustering  
* Dimensionality Reduction  

### 🟥 Reinforcement Learning
* Learning via rewards  

---

## 🎯 5. Problem Formulation

A correct ML system starts with:

* Features (X)  
* Target (Y)  

⚠️ Poor problem definition leads to poor models.

---

## 🔄 6. Machine Learning Pipeline (Deep Dive)

### 🟦 Step 1: Data Extraction
Sources:
* CSV  
* Databases  
* APIs  

---

### 🟩 Step 2: Data Ingestion

```python
import pandas as pd

# ----------- Load Data -----------
def load_data(path):
    return pd.read_csv(
        path,
        sep=",",
        encoding="utf-8",
        na_values=["NA", "?", ""],
        low_memory=False
    )

# ----------- Basic EDA -----------
def explore_data(df):
    print("\n🔹 Preview:")
    print(df.head())

    print("\n🔹 Info:")
    print(df.info())

    print("\n🔹 Summary:")
    print(df.describe(include="all"))

    print("\n🔹 Missing Values:")
    print(df.isnull().sum())


# ----------- Run -----------
if __name__ == "__main__":
    df = load_data("data/dataset.csv")
    explore_data(df)
```

---

### 🟨 Step 3: Data Understanding

```python
import pandas as pd

# ----------- Data Quality Checks -----------
def check_data_quality(df):
    print("\n🔹 Missing Values:")
    print(df.isnull().sum())

    print("\n🔹 Duplicate Rows:")
    print("Duplicates:", df.duplicated().sum())

    print("\n🔹 Cardinality (Unique Values per Column):")
    print(df.nunique())


# ----------- Run -----------
if __name__ == "__main__":
    df = pd.read_csv("data/dataset.csv")
    check_data_quality(df)
```

---

### 🟥 Step 4: Data Cleaning

```python
import pandas as pd

# ----------- Data Cleaning -----------
def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()

    # Fill missing values (numeric → median)
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    # Fill missing values (categorical → mode)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


# ----------- Run -----------
if __name__ == "__main__":
    df = pd.read_csv("data/dataset.csv")
    df = clean_data(df)

    print("✅ Data cleaned successfully")
```

---

### 🔵 Step 5: Feature Engineering

```python
import pandas as pd

# ----------- Encoding & Split -----------
def prepare_data(df, target_col):
    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    # Split features & target
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    return X, y


# ----------- Run -----------
if __name__ == "__main__":
    df = pd.read_csv("data/dataset.csv")
    X, y = prepare_data(df, "target")

    print("✅ Data prepared successfully")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
```

---

### 🧠 Production Insight
🔹 Use ETL Pipelines
Structure your workflow into clear stages:

Extract → Load raw data from source
Transform → Clean, preprocess, and engineer features
Load → Store processed data for modeling

This keeps your pipeline modular and scalable.

🔹 Version Data (DVC)
Use tools like DVC (Data Version Control) to:

Track dataset changes
Reproduce experiments
Manage large files outside Git

Ensures reproducibility and collaboration.

🔹 Validate Data Automatically
Add automated checks to ensure data quality:

Schema validation (column types, formats)
Missing value thresholds
Range checks & constraints

You can use tools like Great Expectations or write custom validation scripts. 

---

## 🧰 7. Libraries & Tools

* NumPy → Numerical computing  
* pandas → Data handling  
* scikit-learn → ML models  
* Matplotlib → Visualization  

---

## 📊 8. Dataset Understanding (EDA)

```python
import matplotlib.pyplot as plt

# ----------- Visualization -----------
def plot_histograms(df):
    df.hist(figsize=(12, 10))
    plt.tight_layout()
    plt.show()


# ----------- Run -----------
if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("data/dataset.csv")
    plot_histograms(df)
```

---

## ⚙️ 9. Model Implementation

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ----------- Train & Evaluate -----------
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\n🔹 Classification Report:")
    print(classification_report(y_test, preds))

    return model


# ----------- Run -----------
if __name__ == "__main__":
    # Assume X, y already prepared
    model = train_model(X, y)
```

---

## 🔗 10. Scikit-learn Pipeline (Advanced)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ----------- Pipeline -----------
def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])


# ----------- Train -----------
if __name__ == "__main__":
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    print("✅ Pipeline trained successfully")
```

---

# 📚 11. Resources

* NumPy → https://numpy.org/doc/
* Pandas → https://pandas.pydata.org/docs/
* Scikit-learn → https://scikit-learn.org/stable/

---

## ⭐ Final Note

This repository is designed to:

* Build strong ML fundamentals  
* Provide hands-on implementation  
* Bridge theory and real-world systems  

---

## 🚀 Future Scope

* Deep Learning (CNN, RNN, LSTM)  
* Deployment (FastAPI, Docker)  
* MLOps (CI/CD, Monitoring)  
