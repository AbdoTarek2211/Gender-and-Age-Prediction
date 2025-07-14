# 🧑‍👩‍👧‍👦 Gender & Age Prediction Web App

This project is a web-based application that predicts **age** and **gender** from a single face image using a deep learning model trained on the **UTK Face Dataset**. It leverages **ResNet50** for transfer learning and is built using **Streamlit** for a clean, interactive user experience.

## 🚀 Live Demo

> Coming Soon (or deploy locally via Streamlit - see below)

---

## 🧠 Model Overview

- **Architecture:** ResNet50 (pretrained, fine-tuned)
- **Task 1:** Gender Classification (Binary: Male/Female)
- **Task 2:** Age Prediction (Regression: 0–100+)
- **Dataset:** [UTKFace](https://susanqq.github.io/UTKFace/)
- **Performance:**
  - **Gender Accuracy:** ~90%
  - **Age MAE (Mean Absolute Error):** ~6 years

---

## 🧪 Try It Yourself

### 🔧 Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/age-gender-predictor.git
cd age-gender-predictor
