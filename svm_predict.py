###########################################
#                                         #
# Fred Sena                               #
#                                         #
# Predict winning record using NFL stats  #
#                                         #
# ML Algorithm: SVM                       #
#                                         #
###########################################

############
# Features #
############

# Features for: basic_nfl_stats.txt

# 1. Points scored (Season)
# 2. Points allowed (Season)
# 3. Sacks (Season)
# 4. Sacks allowed (Season)
# 5. Turnover Ratio (Season)
# 6. Label: 1 if team record > .500; 0 if the team was <= to .500)

# These stats were taken from the 2017 season on NFL.com
# There are 32 instances (One instance for each team in the NFL)
# (32, 6) Matrix

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn import preprocessing


def get_pca(X):

    pca = PCA(n_components=0.95)
    
    new_x = pca.fit_transform(X)

    print new_x.shape

    return new_x



def predict_winning_record(X, Y, split):
    
    classifier = SVC(kernel='poly')

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

filelist = ['basic_nfl_stats.txt', 'winning_record_stats.txt']

data = np.loadtxt(filelist[1], delimiter=",") # Load the data in 

X = data[:, 0:-1]
Y = data[:, -1]


X, Y = shuffle(X, Y)

X = np.delete(X, 26, 1)
print X.shape

X = preprocessing.scale(X) # Scale the data

#################################
# Use PCA to reduce the dataset #
#################################

X = get_pca(X)



predict_winning_record(X, Y, 27) # Call the function

