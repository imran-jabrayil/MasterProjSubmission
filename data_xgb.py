import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
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

# Convert class weights to a scale for XGBoost
total_samples = len(y_train_balanced)
class_weights_xgb = {cls: weight * total_samples / (len(y_train_balanced) * len(class_weights_dict)) for cls, weight in class_weights_dict.items()}

# Build and train the XGBoost model
clf = XGBClassifier(random_state=42, scale_pos_weight=class_weights_xgb[1])  # Adjust for binary classification
clf.fit(X_train_balanced, y_train_balanced)

# Evaluate the model
y_pred = clf.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# SHAP analysis
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# Visualize SHAP summary plot
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=X.columns)
