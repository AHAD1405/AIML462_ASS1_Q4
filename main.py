import random
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import os
import warnings

warnings.filterwarnings("ignore")

# load dataset 
def load_data(column_file_name, data_file_name):
    """
        Extract data from file and return datset
    """
    # Read file and extract data file
    full_path = os.path.abspath(__file__) # Get the full path of the script     
    script_directory = os.path.dirname(full_path) # Get the directory of the script
    data_file = os.path.join(script_directory,data_file_name) 
    columns_file = os.path.join(script_directory, column_file_name) # wbcd.names , sonar.names

    columns_name = []

    # Reading Column names from file
    with open(columns_file,'r') as file_: 
        columns = file_.readlines()
        for idx, line in enumerate(columns): # extract values
            if idx == 0: 
                columns_name.append('target_')   # add column for target column
                continue
            x = line.split()
            columns_name.append(x[0])
            #column_data.append(x[1])

    # Reading dataset from file
    with open(data_file,'r') as data_: 
        data = data_.readlines()
        dataset = pd.DataFrame(columns=columns_name)
        for line in data:
            x = line.split(',')
            dataset.loc[len(dataset)] = x

    # EDA: format Target feature and class feature
    dataset['target'] = [row[0:4] if row[0:4] == 'MUSK' else row[0:8] for row in dataset['target_'].str.strip()]

    # Convert Target columns to numeric
    dataset['class'] = [1 if row[0:4] == 'MUSK' else 0 for row in dataset['target_'].str.strip()]
    
    # Drop unnrelevent columns
    dataset = dataset.drop(columns=['target_'])
    dataset = dataset.drop(columns=['f166:'])
    dataset = dataset.drop(columns=['molecule_name:'])
    dataset = dataset.drop(columns=['conformation_name:'])

    return dataset

def WrapperGA(X, y):
    """
        -  In wrapper methods, we try to use a subset of features and train a model using them. Based on the inferences that we draw from the previous model, 
                we decide to add or remove features from the subset.

        -  Recursive Feature Elimination with Cross-Validated (RFECV) feature selection technique selects the best subset 
                of features for the estimator by removing 0 to N features iteratively using recursive feature elimination.
        
        -  Then it selects the best subset based on the accuracy or cross-validation score or roc-auc of the model. 
                Recursive feature elimination technique eliminates n features from a model by fitting the model multiple times and at each step, removing the weakest features.
        
        -  
    """

    #min_features_to_select = 1  # Minimum number of features to consider
    clf = LogisticRegression()
    cv = StratifiedKFold(5)

    rfecv = RFECV(
        estimator=clf,
        step=1,
        cv=cv,
        scoring="accuracy",
        #min_features_to_select=min_features_to_select,
        n_jobs=2,
    )
    rfecv.fit(X, y)
    predictions = rfecv.predict(X)  # Predictions of the best subset of features

    #correct_predictions = sum(predictions == y)  # Number of correct predictions
    #error_rate = 1 - correct_predictions / len(y)  # Error rate

    selected_features = X.columns[rfecv.support_]  # Names of selected features
    #selected_feature_ratio = len(selected_features) / len(X.columns)  # Ratio of selected features

    # Get the accuracy of the best subset of features
    #accuracy = accuracy_score(y, rfecv.predict(X))
    
    return selected_features

def fitness_func(individual, dataset, target):
    
    all_feature = dataset.shape[1]

    selected_features = [index for index in range(len(individual)) if individual[index] == 1]
    if len(selected_features) == 0:
        return 0
    
    X_selected = dataset.iloc[:, selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, target, test_size=0.3, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_selected)
    
    # Objective 1: Error rate
    correct_predictions = sum(predictions != target)  # Number of correct predictions
    erro_rate = 1 - correct_predictions / len(target)  # Error rate
    
    # Objective 2: Feature selected ratio
    feature_selected_ratio = len(selected_features) / all_feature
    #accuracy = accuracy_score(y_test, predictions) # Accuracy
    
    return round(erro_rate, 4), feature_selected_ratio

# Initialize population
def initialize_population(size, n_features, seed_val):
    np.random.seed(seed_val)
    return [np.random.randint(0, 2, n_features).tolist() for _ in range(size)]

def main():
    # Parameters
    POPULATION_SIZE = 50
    GENERATIONS = 100
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.9
    RUNs = 3 
    SEED_VAL = [20, 40, 60]
    TURNAMENT_K = 2
    
    data_file =  ['clean1.data']    # wbcd.data  , sonar.data
    column_file = ['clean1.names']  # wbcd.names , sonar.names
    
    for idx, datafile_ in enumerate(data_file):
        # Load dataset 
        dataset = load_data(column_file[idx], datafile_)
        Feature = dataset.iloc[:, :-2]
        Target = dataset.iloc[:, -1]

        for run in range(RUNs):
            print('Run:', run)
            fs = WrapperGA(Feature, Target)
            fs_datset = dataset[:][list(fs.array)]
            # Evolutionary algorithm
            population = initialize_population(POPULATION_SIZE, fs_datset.shape[1], SEED_VAL[run])
            #
if __name__ == "__main__":
    main()