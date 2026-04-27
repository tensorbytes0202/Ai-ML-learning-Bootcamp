# 🚀 AI & Machine Learning Bootcamp Repository

> A structured, beginner-to-intermediate guide to understanding and building Machine Learning systems from scratch.

---

# 🧭 Table of Contents

* [1. Introduction to Artificial Intelligence](#-1-introduction-to-artificial-intelligence)
* [2. Introduction to Machine Learning](#-2-introduction-to-machine-learning)
* [3. Current State of AI](#-3-current-state-of-ai)
* [4. Types of Machine Learning](#-4-types-of-machine-learning)
* [5. Problem Formulation](#-5-problem-formulation)
* [6. Machine Learning Pipeline](#-6-machine-learning-pipeline)
* [7. Libraries & Tools](#-7-libraries--tools)
* [8. Dataset Overview](#-8-dataset-overview)
* [9. Implementation Walkthrough](#-9-implementation-walkthrough)
* [10. Scikit-learn Pipeline](#-10-scikit-learn-pipeline)
* [11. Resources & Documentation](#-11-resources--documentation)
* [12. About AI Club KIET](#-12-about-ai-club-kiet)

---

# 🧠 1. Introduction to Artificial Intelligence

## 📌 Formal Definition

Artificial Intelligence (AI) is the study of intelligent agents — systems that:

* Perceive their environment
* Make decisions
* Act to maximize a goal

## 🌍 Real-World Examples

* Self-driving cars 🚗
* Virtual assistants (Siri, Alexa) 🎙️
* Recommendation systems (Netflix, Amazon) 🎯

---

# 🤖 2. Introduction to Machine Learning

Machine Learning (ML) is a subset of AI that allows systems to **learn from data instead of being explicitly programmed**.

## 🔑 Key Idea

> Instead of writing rules → we train models using data.

## 🧩 ML as a Path to Intelligence

ML enables:

* Pattern recognition
* Prediction
* Decision-making

---

# 🌐 3. Current State of AI

Today’s AI ecosystem includes:

### 🤖 AI Agents

Autonomous systems capable of decision-making.

### 🧠 Large Language Models (LLMs)

Models like ChatGPT that understand and generate human language.

### 🎨 Generative AI

* Image generation
* Text generation
* Code generation

⚠️ **Why this matters?**
Understanding ML is essential to work with these technologies.

---

# 📊 4. Types of Machine Learning

## 🟦 Supervised Learning

Model learns from labeled data.

### Types:

* **Regression** → Predict continuous values
* **Classification** → Predict categories

---

## 🟩 Unsupervised Learning

Model finds patterns in unlabeled data.

### Types:

* Clustering
* Dimensionality Reduction

---

# 🎯 5. Problem Formulation

Before building any ML model, define:

### 📥 Input (Features)

Example:

* Age
* Salary

### 📤 Output (Target)

Example:

* Purchased (Yes/No)

### 🎯 Objective

Learn a function:
f(X) → Y

---

# 🔄 6. Machine Learning Pipeline

A real-world ML system is not just a model — it's a pipeline.

## 📌 Steps:

### 1. Data Extraction

Collect raw data from sources.

### 2. Data Ingestion

Load data into system (CSV, DB, APIs).

### 3. Data Transformation

Convert data into usable format.

### 4. Data Preprocessing

* Handling missing values
* Scaling
* Encoding

### 5. Model Training

Train algorithm on data.

### 6. Model Testing

Test on unseen data.

### 7. Model Selection

Choose best model.

### 8. Validation

Ensure model generalizes well.

### 9. Evaluation

Metrics like:

* Accuracy
* Precision
* Recall

---

# 🧰 7. Libraries & Tools

| Library      | Purpose              |
| ------------ | -------------------- |
| NumPy        | Numerical operations |
| Pandas       | Data manipulation    |
| Seaborn      | Data visualization   |
| Matplotlib   | Plotting             |
| Scikit-learn | ML models & pipeline |

---

# 📂 8. Dataset Overview

Dataset used: `/data/dataset.csv`

### Example Columns:

* `age` → User age
* `salary` → Income
* `purchased` → Target variable

---

# 💻 9. Implementation Walkthrough

## 📥 Step 1: Load Data

```python
import pandas as pd
data = pd.read_csv("data/dataset.csv")
```

---

## 🧹 Step 2: Preprocessing

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## 🤖 Step 3: Model Training

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

---

## 📈 Step 4: Evaluation

```python
from sklearn.metrics import accuracy_score

preds = model.predict(X_test)
print(accuracy_score(y_test, preds))
```

---

# 🔗 10. Scikit-learn Pipeline

## ❓ Why Pipeline?

* Clean workflow
* Prevent data leakage
* Easy to scale

---

## ⚙️ Implementation

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)
```

---

# 📚 11. Resources & Documentation

* NumPy → https://numpy.org/doc/
* Pandas → https://pandas.pydata.org/docs/
* Scikit-learn → https://scikit-learn.org/stable/
* Seaborn → https://seaborn.pydata.org/

---

# 🏁 12. About AI Club KIET

## 👥 Who are we?

A student-driven community passionate about Artificial Intelligence and innovation.

## 🎯 Our Goal

* Make AI accessible
* Build real-world projects
* Foster collaboration

## 🏆 Achievements

* Hackathons 🥇
* Research Projects 📄
* Technical Workshops 🎓

---

# 💡 Final Note

This repository is designed to:

* Build strong ML fundamentals
* Provide hands-on experience
* Prepare you for real-world AI systems

---

> ⭐ If you found this helpful, don’t forget to star the repository!
