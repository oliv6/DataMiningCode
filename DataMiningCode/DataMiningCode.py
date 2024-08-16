

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

dataset = pd.read_csv('SaYoPillow.csv')
dataset.head()

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# initialising the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
# learning the statistical parameters for each of the data and transforming
rescaledX = scaler.fit_transform(x)
# summarize transformed data
np.set_printoptions(precision=3)
print(rescaledX[0:,:])

#checking datset is balanced or unbalanced
X_smote, y_smote = SMOTE().fit_resample(x, y)
X_smote = pd.DataFrame(X_smote)
y_smote = pd.DataFrame(y_smote)
print(y_smote.iloc[:, 0].value_counts())

X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size = 0.2, random_state = 0)
print(x)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#NAIVE BAYES

classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred_NB = classifier.predict(X_test)

cm_NB = confusion_matrix(y_test,y_pred_NB)
print(cm_NB)

fig, ax = plot_confusion_matrix(conf_mat=cm_NB)
plt.show()
print(classification_report(y_test, y_pred_NB))

TP_NB=29
TN_NB=19+22+28+28
FP_NB=0
FN_NB=0

Accuracy_NB = (TP_NB + TN_NB) / (TP_NB + TN_NB + FP_NB + FN_NB) 
print(Accuracy_NB)

Precision_NB_class0 =28/28
Precision_NB_class1=19/19
Precision_NB_class2=22/22
Precision_NB_class3=28/28
Precision_NB_class4=29/29
Precision_NB=5/5

Recall_NB_class0=28/28
Recall_NB_class1=19/19
Recall_NB_class2=22/22
Recall_NB_class3=28/28
Recall_NB_class34=29/29
Recall_NB = 5/5
print(Recall_NB)

F1_Score_NB = 2 * Precision_NB * Recall_NB / (Precision_NB + Recall_NB) 
print(F1_Score_NB)
print("\n")

#DECISION TREE

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
print(classifier.fit(X_train, y_train))
y_pred_DT = classifier.predict(X_test)

cm_DT = confusion_matrix(y_test, y_pred_DT) 
print(cm_DT)
fig, ax = plot_confusion_matrix(conf_mat=cm_DT)
plt.show()
print(classification_report(y_test, y_pred_DT))

TP_DT = 29 
TN_DT = 19+1+21+28+28 
FP_DT = 0 
FN_DT = 0 

Accuracy_DT = (TP_DT + TN_DT) / (TP_DT + TN_DT + FP_DT + FN_DT)
print(Accuracy_DT)

Precision_DT_class0 = 28/28
Precision_DT_class1=19/20
Precision_DT_class2=21/21
Precision_DT_class3=28/28
Precision_DT_class4=29/29
Precision_DT=(Precision_DT_class0+Precision_DT_class1+Precision_DT_class2+Precision_DT_class3+Precision_DT_class4)/5
print(Precision_DT)

Recall_DT_class0=28/28
Recall_DT_class1=19/19
Recall_DT_class2=21/22
Recall_DT_class3=28/28
Recall_DT_class4=29/29
Recall_DT = (Recall_DT_class0+Recall_DT_class1+Recall_DT_class2+Recall_DT_class3+Recall_DT_class4)/5
print(Recall_DT)

F1_Score_DT = 2 * Precision_DT * Recall_DT / (Precision_DT + Recall_DT)
print(F1_Score_DT)

#KNN

classifier = KNeighborsClassifier(n_neighbors=5)
print(classifier.fit(X_train, y_train))
y_pred_KNN = classifier.predict(X_test)

cm_KNN=confusion_matrix(y_test, y_pred_KNN)
fig, ax = plot_confusion_matrix(conf_mat=cm_KNN)
plt.show()
print(classification_report(y_test, y_pred_KNN))

TP_KNN=29
TN_KNN=19+22+28+28
FP_KNN=0
FN_KNN=0

Accuracy_KNN = (TP_KNN + TN_KNN) / (TP_KNN + TN_KNN + FP_KNN + FN_KNN) 
print(Accuracy_KNN)

Precision_KNN_class0 =28/28
Precision_KNN_class1=19/19
Precision_KNN_class2=22/22
Precision_KNN_class3=28/28
Precision_KNN_class4=29/29
Precision_KNN=5/5

Recall_KNN_class0=28/28
Recall_KNN_class1=19/19
Recall_KNN_class2=22/22
Recall_KNN_class3=28/28
Recall_KNN_class34=29/29
Recall_KNN = 5/5
print(Recall_KNN)

F1_Score_KNN = 2 * Precision_KNN * Recall_KNN / (Precision_KNN + Recall_KNN) 
print(F1_Score_KNN)


#comparing models

Accuracy = [Accuracy_KNN, Accuracy_DT, Accuracy_NB]
Methods_acc = ['KNN', 'Decision_Trees', 'Naive_Bayes']
Accuracy_pos = np.arange(len(Methods_acc))
plt.ylim(0*.5,2*.5)
plt.bar(Accuracy_pos, Accuracy)
plt.xticks(Accuracy_pos, Methods_acc)
plt.title('Comparing the Accuracy of each model')
plt.show()

Precision = [Precision_DT, Precision_KNN, Precision_NB]
Methods_prec = [ 'Decision_Trees','KNN', 'Naive_Bayes']
precision_position = np.arange(len(Methods_prec))
plt.ylim(0*5,2*.5)
plt.bar(precision_position, Precision)
plt.xticks(precision_position, Methods_prec)
plt.title('Comparing the Precision of each model')
plt.show()

Recall = [Recall_DT, Recall_KNN, Recall_NB]
Methods_rec = [ 'Decision_Trees','KNN', 'Naive_Bayes']
recall_position = np.arange(len(Methods_rec))
plt.ylim(0*.5,2*.5)
plt.bar(recall_position, Recall)
plt.xticks(recall_position, Methods_rec)
plt.title('Comparing the Recall of each model')
plt.show()

F1_Score = [F1_Score_DT, F1_Score_KNN, F1_Score_NB]
Methods_f1 = [ 'Decision_Trees','KNN', 'Naive_Bayes']
f1_score_position = np.arange(len(Methods_f1))
plt.ylim(0*.5,2*.5)
plt.bar(f1_score_position, F1_Score)
plt.xticks(f1_score_position, Methods_f1)
plt.title('Comparing the F-scores of each model')
plt.show()
