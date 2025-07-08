# feature_detective.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('data/train.csv')

# Feature engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Data cleaning/preprocessing (semua fitur yg dipakai harus bebas NaN)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize']
y = df['Survived']

# Baseline (all features)
X = df[features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
baseline_acc = accuracy_score(y_test, y_pred)

# Remove each feature one by one
results = []
for remove_feature in features:
    selected = [f for f in features if f != remove_feature]
    X = df[selected]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append({'Removed': remove_feature, 'Accuracy': acc})

# Show results
print(f'Baseline Accuracy (All Features): {baseline_acc:.4f}')
print('\nAccuracy After Removing Each Feature:')
for row in results:
    print(f"- Remove {row['Removed']}: {row['Accuracy']:.4f}")
