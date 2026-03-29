# Student Stress Level Predictor

A machine learning project that predicts the stress level of a student — **Low**, **Medium**, or **High** — based on their daily lifestyle habits such as sleep, study hours, screen time, and exercise.

---

## Table of Contents

- [About the Project](#about-the-project)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Sample Output](#sample-output)
- [Dataset](#dataset)
- [Model Results](#model-results)

---

## About the Project

Student stress is a growing concern in academic environments. This project uses **Logistic Regression** (a supervised machine learning algorithm) to classify a student's stress level based on 4 measurable input features:

| Feature | Description |
|---|---|
| `sleep_hours` | Hours of sleep per night |
| `study_hours` | Hours spent studying per day |
| `screen_time` | Hours of screen/device usage per day |
| `exercise_hours` | Hours of physical activity per day |

**Target Variable:** `stress_level` → `Low`, `Medium`, or `High`

---

## Tech Stack

- Python 3
- pandas
- scikit-learn

---

## Project Structure

```
stress-predictor/
├── stress_predictor.py   # Main ML script
├── data.csv              # Dataset (300 student records)
└── README.md             # Project documentation
```

---

## Environment Setup

> Make sure you have **Python 3** installed on your system.  
> You can check by running: `python --version` or `python3 --version`

If Python is not installed, download it from: https://www.python.org/downloads/

---

## Installation

**Step 1 — Clone or download the repository**

```bash
git clone https://github.com/palak25bce10920-ship-it/student_stress_predictor

```

**Step 2 — (Optional but recommended) Create a virtual environment**

```bash
python -m venv venv
```

Activate it:

- On Windows:
```bash
venv\Scripts\activate
```

- On Mac/Linux:
```bash
source venv/bin/activate
```

**Step 3 — Install required dependencies**

```bash
pip install pandas scikit-learn
```

---

## How to Run

Make sure `stress_predictor.py` and `data.csv` are in the **same folder**, then run:

```bash
python stress_predictor.py
```

---

## Sample Output

```
Dataset Shape: 300 rows, 5 columns

Training samples : 240
Testing  samples : 60

Model Accuracy: 83.33%

Confusion Matrix:
[[14  2  0]
 [ 1 22  4]
 [ 0  3 14]]

Classification Report:
              precision    recall  f1-score   support

         Low       0.93      0.88      0.90        16
      Medium       0.81      0.81      0.81        27
        High       0.78      0.82      0.80        17

    accuracy                           0.83        60

--- Sample Predictions ---
Student A (sleep=6, study=4, screen=5, exercise=1) → Medium
Student B (sleep=8, study=2, screen=2, exercise=2) → Low
Student C (sleep=4, study=8, screen=7, exercise=0) → High
```

---

## Dataset

The dataset (`data.csv`) contains **300 student records** with the following distribution:

| Stress Level | Count |
|---|---|
| Low | 91 |
| Medium | 119 |
| High | 90 |

---

## Model Results

| Metric | Value |
|---|---|
| Algorithm | Logistic Regression |
| Train/Test Split | 80% / 20% |
| Overall Accuracy | **83.33%** |
| Best Performing Class | Low (F1 = 0.90) |
