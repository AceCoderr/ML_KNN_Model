import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score
import seaborn as sns

# load dataset
col_names = ['x', 'y','out']
dataset = pd.read_csv("dataset2.csv", header=None, names=col_names)
print(dataset.head())
X1=dataset.iloc[:,0]
X2=dataset.iloc[:,1]
X=np.column_stack((X1,X2))
Y=dataset.iloc[:,2]

#split the data
negative_mask = dataset['out'] == -1
positive_mask = dataset['out'] == 1
negative_data = dataset[negative_mask]
positive_data = dataset[positive_mask]
print(negative_data.head())
print(positive_data.head())

negative_X=negative_data.iloc[:,0]
negative_Y=negative_data.iloc[:,1]
positive_X=positive_data.iloc[:,0]
positive_Y=positive_data.iloc[:,1]

#plot the data  
plt.scatter(positive_X, positive_Y, marker='+', label='+1', c='blue')  # + marker for target +1
plt.scatter(negative_X, negative_Y, marker='o', label='-1', c='red')  # o marker for target -1
plt.xlabel('Feature X1')
plt.ylabel('Feature X2')
plt.show()

Model = "KNN"  #change here to run different models

#define list to store the best values
best_parameters = {}
best_scores = {}
best_model = {}
degrees = []

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

best_model1 = None
best_score1 = 0.0
best_degree = 0
optimal_k = 0

mean_score_dict = []
std_dev_dict = []

if(Model == 'LR'):
    degrees = [1,2,3,4,5]  #set the range for degrees
    C_val = [0.001,0.01,0.1,1,10,100]        #Set C values
    #split the dataset into train and test

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    #creating polynomial features from input features
    for d in degrees:
        mean_scores_logreg = []
        std_dev_logreg = [] 
        poly = PolynomialFeatures(degree=d)
        X_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        #Create a logistic regression model
        log_Reg = LogisticRegression(penalty='l2', solver='lbfgs')

        # Create parameter grid to search
        param_grid = {
            'C': C_val,
            'max_iter': [10000]
        }
        #Perform cross validataion
        cvLR = StratifiedKFold(n_splits=5)
        grid_search = GridSearchCV(log_Reg, param_grid, cv=cvLR, scoring='accuracy')
        grid_search.fit(X_poly, y_train)

        # Get the results for mean and standard deviation
        mean_test_scores = grid_search.cv_results_['mean_test_score']
        std_test_scores = grid_search.cv_results_['std_test_score']

        mean_score_dict.append(mean_test_scores)
        std_dev_dict.append(std_test_scores)
        
        # Print the mean and standard deviation for each parameter combination
        for mean_score, std_score, params in zip(mean_test_scores, std_test_scores, grid_search.cv_results_['params']):
            print(f"Mean Test Score: {mean_score:.4f}, Std Test Score: {std_score:.4f}, Parameters: {params}")
            
        # Get the best parameters and best cross-validation score
        best_param = grid_search.best_params_
        best_score = grid_search.best_score_
        best_logReg_model = grid_search.best_estimator_

        best_parameters[d] = best_param
        best_scores[d] = best_score
        best_model[d] = best_logReg_model

        for C in C_val:
            logReg = LogisticRegression(penalty='l2', solver='lbfgs',C=C)
            logReg.fit(X_poly,y_train)
            y_pred = logReg.predict(X_test_poly)

            f1 = f1_score(y_test, y_pred)
            if f1 > best_score1:
                best_score1 = f1
                best_model1 = logReg
                best_degree = d
                
    print(f"Best Score = {best_score1}")
    print(f"Best C value = {best_model1.C}")
    print(f"Best degree = {d}")

elif(Model == 'KNN'):
    #define range for k
    degrees =  list(range(1, 25))
    mean_scores = []
    std_scores = []

    for k in degrees:
        #train Model
        knn = KNeighborsClassifier(n_neighbors=k)

        # Create parameter grid to search
        param_grid = {
            'n_neighbors': [k],
        }

        cv = StratifiedKFold(n_splits=5)
        grid_search = GridSearchCV(knn, param_grid, cv=cv, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # Get the best parameters and best cross-validation score
        best_param = grid_search.best_params_
        best_score = grid_search.best_score_
        best_knn_model = grid_search.best_estimator_

        best_parameters[k] = best_param
        best_scores[k] = best_score
        best_model[k] = best_knn_model

    for k in degrees:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy')
        mean_scores.append(np.mean(scores))
        std_scores.append(np.std(scores))
    # Plot the mean accuracy with error bars for different k values
    plt.errorbar(degrees, mean_scores, yerr=std_scores)
    plt.xlabel('k Values')
    plt.ylabel('Mean Accuracy')
    plt.title('Mean Accuracy with Error Bars for Different k Values')
    plt.xticks(degrees)
    plt.show()
    
    # Find the optimal k value
    optimal_k = degrees[np.argmax(mean_scores)]
    print(f"Optimal k: {optimal_k}")
    print(f"Accuracy for optimal K = {best_scores[optimal_k]}")
else:
    print("wrong model selected. Select LG or KNN")


#plot cross-validation plots
if(Model == 'LR'):
    # Plot mean test scores as a blue line
    for i, d in enumerate(degrees):
        plt.plot(C_val, mean_score_dict[i], 'b', marker='o', label='Mean Test Score')

        plt.errorbar(C_val, mean_score_dict[i], yerr=std_dev_dict[i], fmt='o', capsize=5)

        # Calculate the upper and lower bounds for error bars
        upper_bound = [mean + std for mean, std in zip(mean_score_dict[i], std_dev_dict[i])]
        lower_bound = [mean - std for mean, std in zip(mean_score_dict[i], std_dev_dict[i])]

        # Plot error bars
        plt.fill_between(C_val, upper_bound, lower_bound, color='lightblue', alpha=0.6, label='Mean +/- Std')

        # Set plot labels and title
        plt.xlabel('C Values')
        plt.ylabel('Accuracy')
        plt.title(f'Degree {d}')
        plt.xscale('log')  #Use a logarithmic scale for C values
        plt.legend()
        plt.show()      


#initialize list to store scores, tpr , fpr and confusion matrices
f1_scores = {}
list_tpr = {}
list_fpr = {}
confusion_matrices = {}
roc_curves = {}


if(Model == 'LR'):
    for d in degrees:
        #best logistic implement
        poly = PolynomialFeatures(degree=d)
        X_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        best_logreg_model = best_model[d]
        best_logreg_model.fit(X_poly,y_train)
        y_pred_logreg = best_logreg_model.predict(X_test_poly)
        
        falsePositiveRate, truePositiveRate,_= metrics.roc_curve(y_test,  y_pred_logreg)
        #true positive rate and false positive rate
        list_tpr[d] = truePositiveRate
        list_fpr[d] = falsePositiveRate
        #confusion matrix
        confusion_matrix_logreg = confusion_matrix(y_test, y_pred_logreg)
        confusion_matrices[d] = confusion_matrix_logreg

        #plot predictions vs training data for best estimators for each degree
        plt.scatter(X_test[:, 0], X_test[:, 1], c='green', cmap=plt.cm.Paired, marker='^',  label='Training Data')
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_logreg, cmap=plt.cm.Paired, marker='x',label='Predictions')
        plt.xlabel('Feature X1')
        plt.ylabel('Feature X2')
        plt.title(f'Best Estimator for Degreee = {d}')
        plt.legend()
        plt.show()
    for d in degrees:
        #plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrices[d], annot=True, fmt="d", cmap="Blues", linewidths=0.5, square=True, cbar=False)
        plt.xlabel("Predicted")
        plt.title(f"Cofusion Matrix for degree-{d}")
        plt.ylabel("Actual")
    plt.show()
elif(Model == 'KNN'):
    for d in degrees:
        #best KNN implement
        best_KNN_model = best_model[d]
        best_KNN_model.fit(X_train,y_train)
        y_pred_KNN = best_KNN_model.predict(X_test)

        #ROC curve
        falsePositiveRate, truePositiveRate, _ = metrics.roc_curve(y_test,  y_pred_KNN)
        #true positive rate and false positive rate
        list_tpr[d] = truePositiveRate
        list_fpr[d] = falsePositiveRate
        #confusion matrix
        confusion_matrix_KNN = confusion_matrix(y_test, y_pred_KNN)
        confusion_matrices[d] = confusion_matrix_KNN
    #plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrices[optimal_k], annot=True, fmt="d", cmap="Blues", linewidths=0.5, square=True, cbar=False)
    plt.xlabel("Predicted")
    plt.title(f"Cofusion Matrix for optimal value of K{optimal_k}")
    plt.ylabel("Actual")
    plt.show()


#Adding baseline classifier - strategy - most_frequent
DummyClassifier = DummyClassifier(strategy="most_frequent")
DummyClassifier.fit(X_train,y_train)
y_pred_dummy =DummyClassifier.predict(X_test)

confusion_mat = confusion_matrix(y_test, y_pred_dummy)
score = accuracy_score(y_test,y_pred_dummy)
print(score)
falsePositiveRate_dummy, truePositiveRate_dummy, _ = metrics.roc_curve(y_test,y_pred_dummy)
roc_auc_dummy = metrics.auc(falsePositiveRate_dummy,truePositiveRate_dummy)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrices[d], annot=True, fmt="d", cmap="Blues", linewidths=0.5, square=True, cbar=False)
plt.xlabel("Predicted")
plt.title(f"Cofusion Matrix for baseline classifier - most frequent")
plt.ylabel("Actual")
plt.show()

#plot ROC
plt.figure(figsize=(8, 6))
for d in degrees:
    roc_auc = metrics.auc(list_fpr[d], list_tpr[d])
    #create ROC curve
    plt.plot(list_fpr[d],list_tpr[d],lw=2,label=f"Degree{d}/auc ={roc_auc:.2f}")
    
plt.plot(falsePositiveRate_dummy,truePositiveRate_dummy,lw=2,label=f"auc={roc_auc_dummy:.2f}")

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([-0.01, 1.05])
plt.title(f"ROC curve for {Model}")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()
