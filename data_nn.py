import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import shap
import matplotlib.pyplot as plt

# Load the dataset
dataset_path = 'data.csv'  # Adjust the path if needed
data = pd.read_csv(dataset_path)

# Preprocess dataset
X = data.drop(columns=['fail'])  # 'fail' is the target column
y = data['fail']

# Convert target to categorical for neural network
y = to_categorical(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=data['fail'])

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, data['fail'][X_train.index])
y_train_balanced = to_categorical(y_train_balanced)

# Optionally calculate class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.array(sorted(data['fail'].unique())), y=data['fail'])
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# Build the neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_balanced.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_balanced, y_train_balanced, epochs=50, batch_size=32, class_weight=class_weights_dict, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

print("Confusion Matrix:\n", confusion_matrix(y_test_classes, y_pred_classes))
print("Classification Report:\n", classification_report(y_test_classes, y_pred_classes))




