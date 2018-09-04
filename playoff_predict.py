###########################################
#                                         #
# Fred Sena                               #
#                                         #
# Predict playoff birth using NFL stats   #
#                                         #
# ML Algorithm: Neural Network            #
#                                         #
###########################################


import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.decomposition import PCA


def get_pca(X):

    pca = PCA(n_components=10)
    
    new_x = pca.fit_transform(X)

    print "PCA Shape: ", new_x.shape

    return new_x



def predict_winning_record(X, Y, split):
    
    classifier = MLPClassifier(solver="lbfgs", activation='logistic') #  max_iter=2000

    ####################
    # Cross Validation #
    ####################

    cross_val = cross_val_score(classifier, X, Y, cv=5, n_jobs=-1)


    #######################################
    # Train the model to make predictions #
    #######################################

    x_train = X[0:split] # Use 'split' to split the data
    x_test = X[split:]

    y_train = Y[0:split]
    y_test = Y[split:]

    classifier.fit(x_train, y_train) # Train the model...

    for i in range(len(x_test)):

        predict = classifier.predict(x_test[i].reshape(1,-1)) # Reshape data...

        print " Y:", y_test[i] # Print the original label
        print "Y^:", predict[0] # Print the predicted label
        print 

    print "KFold Cross Validation: \n", cross_val # Print the cross validation scores
    print "\nKFold Average: ", cross_val.mean() # Print the average of the scores
    



###################
# Set up the data #
###################

data = np.loadtxt('advanced_nfl_stats.txt', delimiter=",") # Load the data in 

X = data[:, 0:-1]
Y = data[:, -1]

X = np.delete(X, 26, 1) # Delete Average Time Per Drive

X, Y = shuffle(X, Y)

print X.shape
#################################
# Use PCA to reduce the dataset #
#################################

X = get_pca(X)

predict_winning_record(X, Y, 27) # Call the function
