import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


# Load the pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Load the dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

# Split the dataset into training and testing sets (same as in training)
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, 
    test_size=0.2,       # 20% of the data will be used for testing
    shuffle=True,        # Shuffle the data to avoid any bias
    stratify=labels      # Maintain the same label distribution in both train and test sets
)

# Predict the labels for the test set
y_pred = model.predict(x_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # Weighted for multiclass
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

cm = confusion_matrix(y_test, y_pred, labels=np.unique(labels))

# Display the confusion matrix as a heatmap
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(labels))
fig, ax = plt.subplots(figsize=(12, 12))  # Increase figure size
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax)

# Display the results
print("Results for Gesture Recognition Model")
print("-------------------------------------")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

# Customize the plot
plt.title("Confusion Matrix for Gesture Recognition")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()

# Show the plot
plt.show()
