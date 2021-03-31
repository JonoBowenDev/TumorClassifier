
'''

2020 

Scaffolding code for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.

You are welcome to use the pandas library if you know it.


'''
import numpy as np

from sklearn import model_selection, neighbors, svm, tree
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV 
from sklearn.metrics import confusion_matrix

import pylab as pl

import pandas as pd

import csv

import warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import Sequential, layers

import matplotlib.pyplot as plt
    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [(10183868, "Jonathan", "Bowen")]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DecisionTree_classifier(X_training, y_training):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    # Find the best max depth hyperparameter
    max_depths = [x for x in range(1, 50)] 
    best_max_depth, MSE = get_best_hyperparameter(max_depths, "decision_tree", X_training, y_training)

    print("Best max depth")
    print(best_max_depth)

    # Plot the cross validation process 
    create_plot(max_depths, MSE, "Max Depth", "Misclassification Error", "Decision Tree Cross-validation")

    # Return the best model
    return tree.DecisionTreeClassifier(max_depth=best_max_depth).fit(X_training, y_training)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NearrestNeighbours_classifier(X_training, y_training):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    # Find the best k hyperparameter
    neighbours = [x for x in range(1, 100) if x % 2 != 0]
    optimal_k, MSE = get_best_hyperparameter(neighbours, "nearest_neighbours", X_training, y_training)

    print("Best K")
    print(optimal_k)

    # Plot the cross validation process 
    create_plot(neighbours, MSE, "Number of Neighbours K", "Misclassification Error", "Nearest Neighbours Cross-validation")

    # Return the best model
    return neighbors.KNeighborsClassifier(n_neighbors = optimal_k).fit(X_training, y_training)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SupportVectorMachine_classifier(X_training, y_training):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    # Find the best k hyperparameter
    C_values = [x for x in range(1, 50)]
    optimal_C, MSE = get_best_hyperparameter(C_values, "support_vector_machine", X_training, y_training)

    print("Best C")
    print(optimal_C)

    # Plot the cross validation process 
    create_plot(C_values, MSE, "C value", "Misclassification Error", "Support Vector Cross-validation")

    # Return the best model
    return svm.SVC(C=optimal_C).fit(X_training, y_training)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NeuralNetwork_classifier(X_training, y_training):
    '''  
    Build a Neural Network classifier (with two dense hidden layers)  
    based on the training set X_training, y_training.
    Use the Keras functions from the Tensorflow library

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    # Define training options (reduce epochs to speed up training time) 
    epochs = 10000
    cv_epochs = 1000 # cross validation epochs 
    batch_size = 20

    # Define min and max number of neurons to try as hyperparameter 
    min_neurons = 1
    max_neurons = 30

    model = KerasClassifier(build_fn=create_nn_model, epochs=cv_epochs, batch_size=batch_size, initial_epoch=0, verbose=0)
    # define the grid search parameters
    neurons=[x for x in range(min_neurons, max_neurons)]

    # Perform cross fold validaton to determine best hyperparameters 
    param_grids = dict(n_neurons=neurons)
    grid = GridSearchCV(estimator=model, param_grid=param_grids, n_jobs=-1, cv=5)
    grid_result = grid.fit(X_training, y_training)

    # Print results to the console 
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    best_param = grid_result.best_params_.get('n_neurons')
  
    model = create_nn_model(best_param)

    model.fit(X_training, y_training, epochs = epochs, batch_size = batch_size) 

    return model
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
## AND OTHER FUNCTIONS TO COMPLETE THE EXPERIMENTS

def perform_nearest_neighbours_experiment(X_train, X_test, y_train, y_test):
    '''  
    Trains a k-nearest-neighbours classifier and tests it, generating a confusion chart and recording the accuracy.    

    @param 
	X_train: An array representing the training x values 
    X_test: An array representing the testing x values 
    y_train: An array representing the training y values
    y_test: An array representing the testing y values 

    @return
    accuracy: An integer representing the accuracy of the model

    '''

    print("\n\nNearest Neighbours: \n")

    # Build the classifier and make predictions on testing data
    KNN = build_NearrestNeighbours_classifier(X_train, y_train)
    y_pred = KNN.predict(X_test)

    # Plot a confusion chart of the accuracy 
    create_confusion_chart(y_test, y_pred, "Predicted", "Actual", "Nearest-Neighbours Accuracy: " + str(round(KNN.score(X_test, y_test), 3)))

    # Print some stats to the console 
    accuracy = KNN.score(X_test, y_test)
    print("Accuracy: ")
    print(accuracy)
    print("\n\n")

    return(accuracy)

def perform_support_vector_experiment(X_train, X_test, y_train, y_test):
    '''  
    Trains a support vector machine classifier and tests it, generating a confusion chart and recording the accuracy.    

    @param 
	X_train: An array representing the training x values 
    X_test: An array representing the testing x values 
    y_train: An array representing the training y values
    y_test: An array representing the testing y values 

    @return
    accuracy: An integer representing the accuracy of the model

    '''

    print("\n\nSupport Vector: \n") 

    # Build the classifier and make predictions on testing data 
    SVM = build_SupportVectorMachine_classifier(X_train, y_train)
    y_pred = SVM.predict(X_test)

    # Plot a confusion chart of the accuracy 
    create_confusion_chart(y_test, y_pred, "Predicted", "Actual", "Support Vector Machine Accuracy: " + str(round(SVM.score(X_test, y_test), 3)))

    # Print some stats to the console 
    accuracy = SVM.score(X_test, y_test)
    print("Accuracy: ")
    print(accuracy)
    print("\n\n")

    return accuracy

def perform_decision_tree_experiment(X_train, X_test, y_train, y_test):
    '''  
    Trains a decision tree classifier and tests it, generating a confusion chart and recording the accuracy.    

    @param 
	X_train: An array representing the training x values 
    X_test: An array representing the testing x values 
    y_train: An array representing the training y values
    y_test: An array representing the testing y values 

    @return
    accuracy: An integer representing the accuracy of the model

    '''
    print("\n\nDecision Tree: \n") 

    # Build the classifier and make predictions on testing data 
    DT = build_DecisionTree_classifier(X_train, y_train)
    y_pred = DT.predict(X_test)

    # Plot a confusion chart of the accuracy 
    create_confusion_chart(y_test, y_pred, "Predicted", "Actual", "Decision Tree Accuracy: " + str(round(DT.score(X_test, y_test), 3)))

    # Print some stats to the console 
    accuracy = DT.score(X_test, y_test)
    print("Accuracy: ")
    print(accuracy)
    print("\n\n")

    return accuracy

def perform_neural_network_experiment(X_train, X_test, y_train, y_test):
    '''  
    Trains a neural network classifier and tests it, generating a confusion chart and recording the accuracy.    

    @param 
	X_train: An array representing the training x values 
    X_test: An array representing the testing x values 
    y_train: An array representing the training y values
    y_test: An array representing the testing y values 

    @return
    accuracy: An integer representing the accuracy of the model

    '''

    print("\n\nNeural Network: \n")

    # Build the classifier and make predictions on testing data  
    NN = build_NeuralNetwork_classifier(X_train, y_train)
    test_loss, accuracy = NN.evaluate(X_test, y_test)
    y_pred = NN.predict(X_test)

    # Convert to binary
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0

    # Plot a confusion chart of the accuracy
    create_confusion_chart(y_test, y_pred, "Predicted", "Actual", "Neural Network Accuracy 1: " + str(round(accuracy, 3)))

    # Print some stats to the screen 
    print("Accuracy: ")
    print(accuracy)
    print("\n\n")

    return accuracy
    
def compare_classifiers(KNN_acc, SVM_acc, DT_acc, NN_acc):
    '''  
    Generates a bar graph comparing the accuracy of four different classifiers.    

    @param 
	KNN_acc: The accuracy of the k-nearest-neighbours classifier
    SVM_acc: The accuracy of the support vector machine classifier
    DT_acc: The accuracy of the decision tree classifier
    NN_acc: The accuracy of the neural network classifier

    '''
    height = [KNN_acc, SVM_acc, DT_acc, NN_acc]
    bars = ("KNN", "SVM", "DT", "NN")
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height, color = ("#4fbeff", "#8cd5ff", "#b3e3ff", "#d1eeff"))
    plt.xticks(y_pos, bars)
    plt.title("Comparison of Classifiers")
    plt.xlabel("Classifier")
    plt.ylabel("Accuracy")
    plt.show()

def create_plot(x, y, x_lab, y_lab, title):
    '''  
    Creates a plot with a given title and axis labels. 

    @param 
	x: An array of x values for plotting
	y: An array of y values for plotting 
    x_lab: A string representing the title for the x-axis
    y_lab: A string representing the title for the y-axis 
    title: A string representing the title for the plot 

    '''
    plt.figure(title)
    plt.plot(x, y)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)
    plt.show()

def create_confusion_chart(y_actual, y_pred, x_lab, y_lab, title):
    '''  
    Creates a confusion chart with a given title and axis labels. 

    @param 
	y_actual: An array of values for plotting, representing the actual y values 
	y_pred: An array of values for plotting, representing the predicted y values  
    x_lab: A string representing the title for the x-axis
    y_lab: A string representing the title for the y-axis 
    title: A string representing the title for the plot 

    '''
    cm = confusion_matrix(y_actual, y_pred)
    pl.matshow(cm)
    pl.colorbar()
    pl.title(title)
    pl.xlabel(x_lab)
    pl.ylabel(y_lab)
    pl.show()

def create_nn_model(n_neurons = 1):
    '''  
    Creates a neural network with two dense hidden layers with a given number of neurons.  

    @param 
	n_neurons: An integer representing the number of neurons in the each of the two hidden layers 

    @return
    model: A neural network object that has been compiled   

    '''
    model = Sequential([
        layers.Flatten(input_shape=(29, )),
        layers.Dense(n_neurons, activation="relu"),
        layers.Dense(n_neurons, activation="relu"),
        layers.Dense(1, activation="sigmoid"), 
    ])

    model.compile(optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"])
    
    return model

def get_best_hyperparameter(possible_hyperparameters, model, X_training, y_training):
    '''  
    Finds the best hyperparameter for a given model by performing cross-fold validation.   

    @param 
	possible_hyperparameters: An array representing the possible hyperparameters for the model
    model: A string representing the given model / classifier 
    X_training: An array representing the training x values 
    y_training: An array representing the training y values

    @return
    optimal_h: An integer representing the best hyperparameter for the given model
    MSE: An array representing the errors of the different hyperparameters

    '''

    cv_scores = []

    for h in possible_hyperparameters:
        if model == "nearest_neighbours":
            modelington = neighbors.KNeighborsClassifier(n_neighbors = h)

        elif model == "decision_tree":
            modelington = tree.DecisionTreeClassifier(max_depth=h)

        else: 
            modelington = svm.SVC(C=h)
        
        scores = cross_val_score(modelington, X_training, y_training, cv=10, scoring="accuracy")
        cv_scores.append(scores.mean())
    
    # Find the best h value
    MSE = [1-x for x in cv_scores]
    optimal_h_index = MSE.index(min(MSE))
    optimal_h = possible_hyperparameters[optimal_h_index]

    # Return the optimal h and the all MSE's for plotting  
    return optimal_h, MSE

def prepare_dataset(dataset_path):
    '''  
    Read a comma separated text file where 
	- the first field is a ID number 
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    '''

    X = []
    Y = []

    with open(dataset_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Get the diagnosis and add it to the Y array
            if(row[1] == "B"):
                Y.append(0)
            else:
                Y.append(1)

            # Get the other data and add it to the X array
            X.append(row[2:31])
    
    return np.array(X).astype(np.float), np.array(Y)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    # Write a main part that calls the different 
    # functions to perform the required tasks and repeat your experiments.
    # Call your functions here

    # My Team 
    print(my_team())

    ## EXPERIMENTS

    # Supress warnings for neater output 
    warnings.filterwarnings("ignore")

    # Prepare the dataset 
    X, y = prepare_dataset("medical_records.data")

    # Split into training and testing data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

    # Nearest Neighbours
    test_acc_KNN = perform_nearest_neighbours_experiment(X_train, X_test, y_train, y_test)

    # Support Vector
    test_acc_SVM = perform_support_vector_experiment(X_train, X_test, y_train, y_test)
    
    # Decision Tree
    test_acc_DT = perform_decision_tree_experiment(X_train, X_test, y_train, y_test)
    
    # Neural Network 
    test_acc_NN = perform_neural_network_experiment(X_train, X_test, y_train, y_test)

    # Compare the networks
    compare_classifiers(test_acc_KNN, test_acc_SVM, test_acc_DT, test_acc_NN)

    


