# 🚀 AI & Machine Learning Bootcamp Repository  
> A complete, structured, and deeply explained guide to understanding Machine Learning — from intuition to implementation.

---

# 🧭 Table of Contents
- [1. Introduction to Artificial Intelligence](#-1-introduction-to-artificial-intelligence)
- [2. Introduction to Machine Learning](#-2-introduction-to-machine-learning)
- [3. Current State of AI](#-3-current-state-of-ai)
- [4. Types of Machine Learning](#-4-types-of-machine-learning)
- [5. Problem Formulation](#-5-problem-formulation)
- [6. Machine Learning Pipeline](#-6-machine-learning-pipeline-deep-dive)
- [7. Libraries & Tools](#-7-libraries--tools)
- [8. Dataset Understanding](#-8-dataset-understanding)
- [9. Step-by-Step Implementation](#-9-step-by-step-implementation)
- [10. Scikit-learn Pipeline](#-10-scikit-learn-pipeline-advanced)
- [11. Resources](#-11-resources)
- [12. About AI Club KIET](#-12-about-ai-club-kiet)

---

# 🧠 1. Introduction to Artificial Intelligence

## 📌 What is AI (Intuition First)?
Artificial Intelligence is about building systems that can **think, learn, and make decisions like humans**.

Instead of hardcoding rules, we create systems that:
- Observe the environment  
- Learn patterns  
- Take intelligent actions  

---

## 📐 Formal Definition
AI is defined as:

> “The study of intelligent agents that perceive their environment and act upon it to maximize the probability of achieving goals.”

---

## 🔍 Breaking the Definition
- **Agent** → Any system (robot, software, model)  
- **Perception** → Taking input (images, text, data)  
- **Action** → Output or decision  
- **Goal** → Objective (accuracy, prediction, optimization)  

---

## 🌍 Real-Life Examples
- YouTube recommendations  
- Google Maps route optimization  
- Chatbots  

👉 AI is everywhere — and ML is how we build it.

---

# 🤖 2. Introduction to Machine Learning

## 📌 Core Idea
Machine Learning allows systems to **learn patterns from data instead of being explicitly programmed**.

---

## 🧠 Traditional Programming vs ML

| Traditional Programming | Machine Learning |
|------------------------|----------------|
| Rules + Data → Output  | Data + Output → Rules |
| Manual logic           | Learns automatically |

---

## 🎯 Why ML is Powerful
Real-world problems:
- Too complex to hardcode  
- Contain hidden patterns  
- Change over time  

ML adapts using data.

---

## 🧩 ML = Function Learning

f(X) → Y  

Where:
- X = Input (features)  
- Y = Output (target)  

---

# 🌐 3. Current State of AI

## 🤖 AI Agents
Autonomous systems capable of decision-making.

## 🧠 Large Language Models (LLMs)
Models that understand and generate human language.

## 🎨 Generative AI
Models that generate:
- Images  
- Text  
- Code  

⚠️ These systems are built using ML pipelines.

---

# 📊 4. Types of Machine Learning

## 🟦 Supervised Learning

### Definition
Learning from labeled data.

### Types:
- **Regression** → Continuous output  
- **Classification** → Categories  

---

## 🟩 Unsupervised Learning

### Definition
Learning without labels.

### Types:
- Clustering  
- Dimensionality Reduction  

---

# 🎯 5. Problem Formulation

## 📥 Input (Features)
Example:
- Age  
- Salary  

## 📤 Output (Target)
Example:
- Purchased (Yes/No)

## 🎯 Objective

f(X) → Y  

⚠️ If problem formulation is wrong → entire ML pipeline fails.

---

# 🔄 6. Machine Learning Pipeline (Deep Dive)

## Step 1: Data Extraction
Collect data from:
- CSV  
- Databases  
- APIs  

---

## Step 2: Data Ingestion

```python
import pandas as pd
data = pd.read_csv("data/dataset.csv")
