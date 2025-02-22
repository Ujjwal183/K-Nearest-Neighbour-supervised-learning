import pandas as pd
"""
step-1 : creating data frame from 
""" 
dataframe = pd.read_csv('Data/Iris.csv')
# head = dataframe.head()
# print(head)
# print(dataframe);
"""
Step-2 : Removing the label(y) and irrelevant data
        creating the label variable 
"""
label = dataframe['Species']
#axis 0 : for row-wise(default)
#axis 1 : for column-wise 
data = dataframe.drop('Id',axis=1)
# print(data)
data = data.drop('Species',axis=1)
# print(data)
"""
Step-3 : Split the dataset in training and testing
"""
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(data,label,test_size=0.25,random_state=7)

"""
Step-4 : Training the model
"""
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(X_train,Y_train)

""" 
Step-5 : prediction 
"""
Y_pred = classifier.predict(X_test)
# print(Y_pred)
"""
Step-6 : Check the accuracy of the model 
"""
from sklearn import metrics
score = metrics.accuracy_score(Y_test,Y_pred)
print(score)
"""
Step-7 : Plot a confusion Matrix
"""
import matplotlib.pyplot as plt
cm = metrics.confusion_matrix(Y_test,Y_pred)
#display label must be equal to the number of classes of label
cm_display = metrics.ConfusionMatrixDisplay(cm,display_labels=classifier.classes_)
# print(cm_display)
cm_display.plot()
plt.show()