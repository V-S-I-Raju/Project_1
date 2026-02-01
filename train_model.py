# -----------------------------
# Student Performance ML Model
# Using ONLY Scikit-Learn
# -----------------------------

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load Dataset
data = pd.read_csv("student_performance.csv")

# 2. Separate features (X) and target (y)
X = data[["study_hours", "attendance", "sleep_hours", "mobile_hours"]]
y = data["score"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create Model
model = LinearRegression()

# 5. Train Model
model.fit(X_train, y_train)

# 6. Test Model
y_pred = model.predict(X_test)

# 7. Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Training Completed")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# 8. Save Model
joblib.dump(model, "student_performance_model.pkl")

print("Model saved as student_performance_model.pkl")
