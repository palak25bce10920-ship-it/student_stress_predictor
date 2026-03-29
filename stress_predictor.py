import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# 1. LOAD DATA
df = pd.read_csv("data.csv")
print("Dataset Preview (first 5 rows):")
print(df.head())
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
# 2. ENCODE TARGET VARIABLE
# Converting stress level text labels to numbers (ML models require numeric input)
label_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['stress_level'] = df['stress_level'].map(label_map)

# Drop rows where stress_level couldn't be mapped (i.e., unexpected values)
df.dropna(subset=['stress_level'], inplace=True)
df['stress_level'] = df['stress_level'].astype(int)
# 3. DEFINE FEATURES AND TARGET
feature_cols = ['sleep_hours', 'study_hours', 'screen_time', 'exercise_hours']
features = df[feature_cols]   # Input variables
target   = df['stress_level'] # Output variable
# 4. SPLIT DATA (80% train / 20% test)
# random_state=42 ensures same split every run (reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)
print(f"raining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")
# 5. TRAIN THE MODEL
# max_iter=1000 prevents convergence warnings on larger datasets
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 6. EVALUATE THE MODEL
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))
# 7. PREDICT ON NEW / CUSTOM INPUT
reverse_map = {0: "Low", 1: "Medium", 2: "High"}
def predict_stress(sleep, study, screen, exercise):
    """
    Predicts stress level for a student given their daily habits.

    Parameters:
        sleep    (float): Hours of sleep per night
        study    (float): Hours spent studying per day
        screen   (float): Hours of screen time per day
        exercise (float): Hours of exercise per day

    Returns:
        str: Predicted stress level — 'Low', 'Medium', or 'High'
    """
    sample = pd.DataFrame(
        [[sleep, study, screen, exercise]],
        columns=feature_cols
    )
    prediction = model.predict(sample)[0]
    return reverse_map[prediction]
# Example predictions 
print("Sample Predictions")
print(f"Student A (sleep=6, study=4, screen=5, exercise=1) {predict_stress(6, 4, 5, 1)}")
print(f"Student B (sleep=8, study=2, screen=2, exercise=2) {predict_stress(8, 2, 2, 2)}")
print(f"Student C (sleep=4, study=8, screen=7, exercise=0) {predict_stress(4, 8, 7, 0)}")
