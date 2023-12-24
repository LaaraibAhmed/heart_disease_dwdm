import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# loading and reading the dataset
heart = pd.read_csv("heart_cleveland_upload.csv")

# creating a copy of the dataset
heart_df = heart.copy()
heart_df = heart_df.rename(columns={'condition': 'target'})

# model building
x = heart_df.drop(columns='target')
y = heart_df.target

# splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# feature scaling
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.fit_transform(x_test)

# Creating classifiers
classifiers = {
    'SVM': SVC(),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier()
}

# Train and evaluate each classifier
results = {}
for name, model in classifiers.items():
    model.fit(x_train_scaler, y_train)
    y_pred = model.predict(x_test_scaler)
    accuracy = accuracy_score(y_test, y_pred) * 100
    results[name] = accuracy

    print(f'{name} Accuracy: {accuracy:.2f}%')
    print('Classification Report\n', classification_report(y_test, y_pred))
    print('-' * 50)
    
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=heart_df['target'].unique(), yticklabels=heart_df['target'].unique())
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    print('-' * 50)
# Save the best model
best_model_name = max(results, key=results.get)
best_model = classifiers[best_model_name]
filename = f'heart-disease-prediction-{best_model_name.lower().replace(" ", "_")}-model.pkl'
pickle.dump(best_model, open(filename, 'wb'))
print(f"The best model is {best_model_name} with an accuracy of {results[best_model_name]:.2f}%.")
