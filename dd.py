import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Set pandas option to opt-in to future behavior
pd.set_option('future.no_silent_downcasting', True)

# Load and prepare the data
column_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 
                 'A11', 'A12', 'A13', 'A14', 'A15', 'Target']
df = pd.read_csv('crx.csv', header=None, names=column_names)
df = df.replace('?', pd.NA)

# Handle missing values in numerical columns (imputation with mean)
numerical_cols = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce') 
    df[col] = df[col].fillna(df[col].mean())

# Encode categorical features and save mappings
categorical_cols = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']
category_mappings = {}

for col in categorical_cols:
    df[col] = df[col].fillna('missing').astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    category_mappings[col] = le.classes_

# Save the category mappings
joblib.dump(category_mappings, 'category_mappings.pkl')

# Feature scaling for numerical features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Encode the Target variable BEFORE splitting the data
df['Target'] = df['Target'].replace({'+': 1, '-': 0}).astype(int, errors='ignore') 

# Split into training and testing sets
X = df.drop('Target', axis=1)
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the trained model and scaler using joblib
joblib.dump(model, 'credit_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Evaluation (Optional)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
