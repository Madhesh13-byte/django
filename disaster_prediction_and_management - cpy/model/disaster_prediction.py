# 1. Import Libraries
import pandas as pd
import numpy as np
import random
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 2. Load Dataset
file_path = 'F:\\disaster_prediction_and_management\\disaster_dataset.xlsx'
df = pd.read_excel(file_path)

# 3. Select Relevant Columns
columns_needed = ['Disaster_Type', 'Magnitude', 'Depth', 'Wind_Speed', 'Rainfall', 'Temperature', 'Humidity']
df = df[columns_needed]

# 4. Handle Missing Values
for col in ['Magnitude', 'Depth', 'Wind_Speed', 'Rainfall', 'Temperature', 'Humidity']:
    df[col] = df[col].fillna(df[col].mean())

df = df.dropna(subset=['Disaster_Type'])

# 5. Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 6. Split Dataset
X = df[['Magnitude', 'Depth', 'Wind_Speed', 'Rainfall', 'Temperature', 'Humidity']]
y = df['Disaster_Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train the Model
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9. Smart Prediction Function
def smart_predict(event, best_model):
    if event['Rainfall'] > 300 and event['Wind_Speed'] < 180:
        return 'Flood'
    elif event['Wind_Speed'] > 180:
        return 'Hurricane'
    elif event['Temperature'] > 35 and event['Humidity'] < 30 and event['Wind_Speed'] > 20:
        return 'Wildfire'
    else:
        event_df = pd.DataFrame([event])
        return best_model.predict(event_df)[0]

# 10. Predict for a New Event
import pandas as pd
import numpy as np

np.random.seed(42)
data = {
    'Magnitude': np.random.uniform(1.0, 8.0, 100),
    'Depth': np.random.uniform(1.0, 500.0, 100),
    'Wind_Speed': np.random.uniform(5.0, 250.0, 100),
    'Rainfall': np.random.uniform(1.0, 400.0, 100),
    'Temperature': np.random.uniform(18.0, 40.0, 100),
    'Humidity': np.random.uniform(20.0, 95.0, 100)
}

df = pd.DataFrame(data)

predicted_disasters = []
for index, row in df.iterrows():
    event = row.to_dict()
    predicted_disaster = smart_predict(event, best_model)
    predicted_disasters.append(predicted_disaster)

df['Predicted Disaster'] = predicted_disasters
print(df.head(10))

random_index = random.randint(0, X_test.shape[0] - 1)
sample_input = X_test.iloc[random_index].to_dict()
actual_disaster = y_test.iloc[random_index]

predicted_disaster_sample = smart_predict(sample_input, best_model)

print("\nRandom Test Sample Features:", sample_input)
print("Actual Disaster Type:", actual_disaster)
print("Predicted Disaster Type:", predicted_disaster_sample)

importances = best_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8,5))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importances')
plt.show()

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=best_model.classes_,
            yticklabels=best_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

with open('random_forest_disaster_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
joblib.dump(best_model, 'disaster_prediction_model.pkl')