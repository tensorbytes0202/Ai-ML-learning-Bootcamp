# 🚀 AI & Machine Learning Bootcamp Repository  
> A structured, implementation-first guide to Machine Learning — from intuition to production-ready workflows.

---

 🧭 Table of Contents
- [1. Introduction to Artificial Intelligence](#-1-introduction-to-artificial-intelligence)
- [2. Introduction to Machine Learning](#-2-introduction-to-machine-learning)
- [3. Current State of AI](#-3-current-state-of-ai)
- [4. Types of Machine Learning](#-4-types-of-machine-learning)
- [5. Problem Formulation](#-5-problem-formulation)
- [6. Machine Learning Pipeline](#-6-machine-learning-pipeline-deep-dive)
- [7. Libraries & Tools](#-7-libraries--tools)
- [8. Dataset Understanding (EDA)](#-8-dataset-understanding-eda)
- [9. Model Implementation](#-9-model-implementation)
- [10. Scikit-learn Pipeline](#-10-scikit-learn-pipeline-advanced)
- [11. Resources](#-11-resources)

---

# 🧠 1. Introduction to Artificial Intelligence

Artificial Intelligence (AI) refers to systems that can:
- Learn from data  
- Make decisions  
- Adapt over time  

> AI systems are intelligent agents that perceive and act to achieve goals.

---

# 🤖 2. Introduction to Machine Learning

Machine Learning enables systems to learn patterns from data instead of hardcoded rules.

### 🧩 ML as Function Learning

f(X) → Y


- X → Features  
- Y → Target  

---

# 🌐 3. Current State of AI

- 🤖 AI Agents  
- 🧠 Large Language Models  
- 🎨 Generative AI  

⚠️ All modern AI systems rely on ML pipelines.

---

# 📊 4. Types of Machine Learning

### 🟦 Supervised Learning
- Regression  
- Classification  

### 🟩 Unsupervised Learning
- Clustering  
- Dimensionality Reduction  

### 🟥 Reinforcement Learning
- Learning via rewards  

---

# 🎯 5. Problem Formulation

A correct ML system starts with:
- Features (X)  
- Target (Y)  

⚠️ Poor problem definition leads to poor models.

---

# 🔄 6. Machine Learning Pipeline (Deep Dive)

## 🟦 Step 1: Data Extraction
Sources:
- CSV  
- Databases  
- APIs  

---

## 🟩 Step 2: Data Ingestion

```python
import pandas as pd

data = pd.read_csv(
    "data/dataset.csv",
    sep=",",
    encoding="utf-8",
    na_values=["NA", "?", ""],
    low_memory=False
)

print(data.head())
print(data.info())
print(data.describe(include='all'))

📎 Footnote:
read_csv() supports flexible parsing, missing value handling, and memory optimization via parameters like sep, na_values, and low_memory.

🟨 Step 3: Data Understanding
missing = data.isnull().sum()
duplicates = data.duplicated().sum()
cardinality = data.nunique()

print(missing)
print("Duplicates:", duplicates)
print(cardinality)
🟥 Step 4: Data Cleaning
data = data.drop_duplicates()

for col in data.select_dtypes(include='number').columns:
    data[col] = data[col].fillna(data[col].median())

for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].fillna(data[col].mode()[0])
🔵 Step 5: Feature Engineering
data = pd.get_dummies(data, drop_first=True)

X = data.drop("target", axis=1)
y = data["target"]
🧠 Production Insight
Use ETL pipelines
Version data (DVC)
Validate data automatically
🧰 7. Libraries & Tools
NumPy → Numerical computing
pandas → Data handling
scikit-learn → ML models
Matplotlib → Visualization
📊 8. Dataset Understanding (EDA)
import matplotlib.pyplot as plt

data.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()
⚙️ 9. Model Implementation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print(classification_report(y_test, preds))

📎 Footnote:
LogisticRegression minimizes log-loss using optimization algorithms like LBFGS and supports binary and multiclass classification.

🔗 10. Scikit-learn Pipeline (Advanced)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

📎 Footnote:
Pipelines help prevent data leakage, ensure reproducibility, and standardize workflows.

📚 11. Resources
Official documentation:
pandas
scikit-learn
⭐ Final Note

This repository is designed to:

Build strong ML fundamentals
Provide hands-on implementation
Bridge theory and real-world systems
🚀 Future Scope
Deep Learning (CNN, RNN, LSTM)
Deployment (FastAPI, Docker)
MLOps (CI/CD, Monitoring)
