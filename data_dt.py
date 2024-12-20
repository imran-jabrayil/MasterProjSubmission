import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import shap
import matplotlib.pyplot as plt

# Load the dataset
dataset_path = 'data.csv'  # Adjust the path if needed
data = pd.read_csv(dataset_path)

# Preprocess dataset
X = data.drop(columns=['fail'])  # 'fail' is the target column
y = data['fail']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Optionally calculate class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.array(sorted(y.unique())), y=y)
class_weights_dict = {cls: weight for cls, weight in zip(sorted(y.unique()), class_weights)}

# Build and train the decision tree model
clf = DecisionTreeClassifier(random_state=42, class_weight=class_weights_dict, max_depth=5)
clf.fit(X_train_balanced, y_train_balanced)

# Evaluate the model
y_pred = clf.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature importance
feature_importances = clf.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(X.columns, feature_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance of Decision Tree Model')
plt.show()

# SHAP analysis
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

