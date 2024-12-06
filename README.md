import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Example dataset: Sports over/under data
data = pd.DataFrame({
    'team1_avg': [1.5, 2.0, 1.2, 1.8, 2.5],
    'team2_avg': [1.3, 1.7, 1.1, 1.9, 2.0],
    'team1_form': [0.8, 1.0, 0.6, 0.9, 1.2],
    'team2_form': [0.7, 0.9, 0.5, 1.0, 1.1],
    'over_under': [1, 1, 0, 1, 1]  # 1 = Over, 0 = Under
})

# Features and target
X = data[['team1_avg', 'team2_avg', 'team1_form', 'team2_form']]
y = data['over_under']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
# New match data for prediction
new_match = pd.DataFrame({
    'team1_avg': [1.7],
    'team2_avg': [1.6],
    'team1_form': [1.0],
    'team2_form': [0.9]
})

# Predict probabilities
over_prob = model.predict_proba(new_match)[0][1]  # Probability of "Over"
print(f"Probability of Over: {over_prob:.2f}")

# Decision rule
threshold = 0.7  # Confidence threshold
if over_prob > threshold:
    print("Bet on Over!")
else:
    print("Bet on Under!")

