"""CROP RECOMMENDATION SYSTEM BASED ON SOIL AND WEATHER CONDITIONS"""

"""PREPROCESSING"""

"""Importing libraries"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")#ignoring warnings

pd.set_option("display.max_columns",None)

"""csv file reading"""
df = pd.read_csv("Crop_recommendation.csv")

print("First five rows of data set is:\n",df.head())
print("Names of the columns:\n",df.columns)
print("Information about the columns:\n",df.info())
print("Sum of Null values:\n",df.isnull().sum())
print("Statistical info of data:\n",df.describe())

df.drop_duplicates()

print("Unique values of label column:\n",df["label"].unique())

"""EXPLORATORY DATA ANALYSIS"""

"""Co relation Matrix"""
numerical_df = df.drop(["label"],axis=1)
print("Numerical columns:\n",numerical_df)
corr_matrix = numerical_df.corr()
print("Correlation Matrix:\n",corr_matrix)
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix,cmap="coolwarm")
plt.title("Correlation matrix of data")
plt.show()

"""Box Plot"""
plt.figure(figsize=(12,8))
sns.boxplot(df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]])
plt.title("Box plot of Nutrients & Environmental factors")
plt.show()

"""Histogram"""
fig, axes = plt.subplots(3, 3, figsize=(12, 8))
axes = axes.flatten()
columns_to_plot = df.columns[:7]

for i, col in enumerate(columns_to_plot):
    sns.histplot(df[col], bins=20, kde=True, ax=axes[i])
    axes[i].set_title(f"Histogram of {col}")
axes[-2].axis("off")
axes[-1].axis("off")
plt.tight_layout()
plt.show()

"""Bar Plot"""
columns_to_barplot = ["N", "P", "K"]
sum_values = df[columns_to_barplot].sum()
plt.figure(figsize=(12, 8))
plt.bar(sum_values.index, sum_values.values, color=["green", "brown", "red"])
plt.title("Sum of Values for 3 Columns")
plt.xlabel("Nutrients")
plt.ylabel("Values")
plt.show()

"""Pair-plot"""
sns.pairplot(data=df,hue="label")
plt.show()

"""Groupby"""
group_by = df.groupby("label").mean()#.reset_index()
print("Goupby categorical columns:\n",group_by)

"""FEATURE ENGINEERING"""

"""Splitting x and y for model training"""
x = df.iloc[:,:7]
y = df.iloc[:,-1]
print("x value:\n",x.head())
print("y value:\n",y.head())

"""Scaling and encoding Dataframe"""
X = MinMaxScaler().fit_transform(x)
print("Scaled x value:\n",X)
Y = LabelEncoder().fit_transform(y)
print("label encoded y value:\n",Y)

"""Train test splitting X and Y"""
x_train,x_test, y_train, y_test = train_test_split(X,Y, test_size =0.2, random_state=42)

"""MODEL SELECTION, TRAINING, PREDICTION AND EVALUATION"""

"""Logistic regression model"""
Logistic_Regression = LogisticRegression()
Logistic_Regression.fit(x_train,y_train)
prediction = Logistic_Regression.predict(x_test)
print("x TEST:\n",x_test)
print("y TEST:\n",y_test)
print("Logistic Regression prediction:\n",prediction)

"""Logistic regression model accuracy score"""
Logistic_accuracy = accuracy_score(y_test, prediction)
print("Logistic Regression Accuracy:\n",Logistic_accuracy)

"""Logistic regression model classification report"""
log_reg_class_report = classification_report(y_test,prediction)
print("Logistic Regression Classification Report:\n",log_reg_class_report)

"""Logistic regression model confusion matrix"""
confusion_matrix_output =confusion_matrix(y_test,prediction)
print("Logistic Regression Confusion Matrix:\n",confusion_matrix_output)

"""Logistic regression model cross validation"""
log_reg_cv_accuracy = cross_val_score(Logistic_Regression,X,Y,cv=5,scoring='accuracy').mean()
print("Logistic Regression Cross validation Accuracy:\n",log_reg_cv_accuracy)

"""hyper parameter tuning logistic regression model and prediction"""
param_grid_LR = {"C": [0.01, 0.1 ,1, 10 , 100],
              "penalty": ["l1", "l2","elasictnet","none"],
              "solver" : ["liblinear", "sega" , "lbfgs"],
              "max_iter": [100,200,500]
              }
grid_search_LR = GridSearchCV(Logistic_Regression,param_grid_LR, cv=4)
grid_search_LR.fit(x_train,y_train)
best_estimator_LR = grid_search_LR.best_estimator_
prediction_grid_LR = best_estimator_LR.predict(x_test)
print("Logistic Regression Grid search Prediction:\n",prediction_grid_LR)

"""Logistic regression model hyper parameter accuracy"""
accuracy_grid_LR = accuracy_score(y_test, prediction_grid_LR)
print("Logistic Regression Grid search Accuracy:\n",accuracy_grid_LR)

"""Decision Tree Model prediction"""

Decision_tree = DecisionTreeClassifier(random_state=42)
Decision_tree.fit(x_train,y_train)
prediction_DT = Decision_tree.predict(x_test)
print("Decision Tree Prediction:\n",prediction_DT)

"""Decision Tree Model accuracy score"""
accuracy_DT = accuracy_score(y_test, prediction_DT)
print("Decision Tree Accuracy:\n",accuracy_DT)

"""Decision Tree Model classification report"""
dec_tree_class_report = classification_report(y_test,prediction_DT)
print("Decision Tree Classification Report:\n",dec_tree_class_report)

"""Decision Tree Model confusion matrix"""
confusion_matrix_output_DT = confusion_matrix(y_test,prediction_DT)
print("Decision Tree Confusion Matrix:\n",confusion_matrix_output_DT)

"""Decision Tree Model cross validation score"""
dec_tree_cv_accuracy = cross_val_score(Decision_tree,X,Y,cv=5,scoring="accuracy").mean()
print("Decision Tree Cross validation Accuracy:\n",dec_tree_cv_accuracy)

"""hyper parameter tuning logistic regression model and prediction"""
param_grid_DT = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search_DT = GridSearchCV(Decision_tree,param_grid_DT, cv=5)
grid_search_DT.fit(x_train,y_train)
best_estimator_DT = grid_search_DT.best_estimator_
prediction_grid_DT = best_estimator_DT.predict(x_test)
print("Decision Tree Grid Search Prediction:\n",prediction_grid_DT)

"""Decision Tree Model hyper parameter accuracy"""
accuracy_grid_DT = accuracy_score(y_test, prediction_grid_DT)
print("Decision Tree Grid Search Accuracy:\n",accuracy_grid_DT)

"""Random Forest Model prediction"""

Random_Forest = RandomForestClassifier(random_state=42)
Random_Forest.fit(x_train,y_train)
prediction_RF = Random_Forest.predict(x_test)
print("Random Forest Prediction:\n",prediction_RF)

"""Random forest Model accuracy score"""
accuracy_RF = accuracy_score(y_test, prediction_RF)
print("Random Forest Accuracy:\n",accuracy_RF)

"""Random forest Model cross classification report"""
RF_class_report = classification_report(y_test,prediction_RF)
print("Random Forest Classification Report:\n",RF_class_report)

"""Random forest Model confusion matrix"""
confusion_matrix_output_RF = confusion_matrix(y_test,prediction_RF)
print("Random Forest Confusion Matrix:\n",confusion_matrix_output_RF)

"""Random forest Model cross accuracy score"""
RF_cv_accuracy = cross_val_score(Random_Forest,X,Y,cv=5,scoring="accuracy").mean()
print("Random Forest Cross validation Accuracy:\n",RF_cv_accuracy)

"""hyper parameter tuning Random Forest model and prediction"""
param_grid_RF = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10]
}
grid_search_RF = GridSearchCV(Random_Forest,param_grid_RF, cv=5)
grid_search_RF.fit(x_train,y_train)
best_estimator_RF = grid_search_RF.best_estimator_
prediction_grid_RF= best_estimator_RF.predict(x_test)
print("Random Forest Grid Search Prediction:\n",prediction_grid_RF)

"""Random Forest Model hyper parameter accuracy"""
accuracy_grid_RF = accuracy_score(y_test, prediction_grid_RF)
print("Random Forest Grid Search Accuracy:\n",accuracy_grid_RF)

"""Support Vector Model training and prediction"""

Support_vector_classifier = SVC(random_state=42)
Support_vector_classifier.fit(x_train,y_train)
prediction_SV = Support_vector_classifier.predict(x_test)
print("Support Vector Classifier Prediction:\n",prediction_SV)

"""Support vector model Accuracy score"""
accuracy_SV = accuracy_score(y_test, prediction_SV)
print("Support Vector Classifier Accuracy:\n",accuracy_SV)

"""Support vector model Classification report"""
SVC_class_report = classification_report(y_test,prediction_SV)
print("Support Vector Classification Report:\n",SVC_class_report)

"""Support Vector Model confusion matrix"""
confusion_matrix_output_SV = confusion_matrix(y_test,prediction_SV)
print("Random Forest Confusion Matrix:\n",confusion_matrix_output_SV)

"""Support vector cross Validation Accuracy score"""
SV_cv_accuracy = cross_val_score(Support_vector_classifier,X,Y,cv=5,scoring="accuracy").mean()
print("Support Vector Classifier Cross Validation Accuracy:\n",SV_cv_accuracy)

"""hyper parameter tuning Support vector  model and prediction"""
param_grid_SV = {
    "kernel": ["linear", "rbf", "poly"],
    "C": [0.1, 1, 10],
    "gamma": [0.1, 1, 10]
}
grid_search_SV = GridSearchCV(Support_vector_classifier,param_grid_SV, cv=5)
grid_search_SV.fit(x_train,y_train)
best_estimator_SV = grid_search_SV.best_estimator_
prediction_grid_SV = best_estimator_SV.predict(x_test)
print("Support Vector Classifier Grid Search Prediction:\n",prediction_grid_SV)

"""Support Vector Model hyper parameter accuracy"""
accuracy_grid_SV = accuracy_score(y_test, prediction_grid_SV)
print("Support Vector Classifier Grid Search Accuracy:\n",accuracy_grid_SV)

"""plotting and comparing all the models accuracy"""
Models= ["Logistic_Regression", "Decision_tree","Random_Forest","Support_vector_classifier"]
scores= [Logistic_accuracy,accuracy_DT,accuracy_RF,accuracy_SV]
colors = ["red","blue","brown","green"]
plt.figure(figsize=(10,8))
plt.bar(Models,scores,color= colors)
plt.grid(True)
plt.xlabel("Models")
plt.ylabel("Scores")
plt.show()

"""plotting and comparing all the hyper parameter tuned models accuracy """

Models_Grid_search= ["Logistic_Regression_GS", "Decision_tree_GS","Random_Forest_GS","Support_vector_classifier_GS"]
scores= [accuracy_grid_LR,accuracy_grid_DT,accuracy_RF,accuracy_grid_SV]
plt.figure(figsize=(10,8))
plt.bar(Models_Grid_search,scores,color= colors)
plt.grid(True)
plt.xlabel("Models")
plt.ylabel("Scores")
plt.show()
