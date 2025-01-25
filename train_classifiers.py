import pickle

#libraries used to train classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np

data_dict=pickle.load(open('./data.pickle','rb'))

#To check homogenous shape of all data images
#for i, item in enumerate(data_dict['data']):
#    print(f"Item {i}: Type={type(item)}, Length={len(item) if hasattr(item, '__len__') else 'N/A'}")

data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

# Split the dataset into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, 
    test_size=0.2,       # 20% of the data will be used for testing
    shuffle=True,        # Shuffle the data to avoid any bias
    stratify=labels      # Maintain the same label distribution in both train and test sets
)
# Initialize the RandomForestClassifier
model = RandomForestClassifier()

# Train the model using the training data
model.fit(x_train, y_train)

# Predict the labels of the test set
y_predict = model.predict(x_test)

# Calculate the accuracy of the model
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

# Now we need to save this model 
f=open('model.p','wb')
pickle.dump({'model':model},f)
f.close()