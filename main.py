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
from sklearn.preprocessing import MinMaxScaler
import os
import warnings
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
import matplotlib

#matplotlib.use('Agg')  # Use a non-interactive backend
matplotlib.use('TkAgg')  # Use a non-interactive backend
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)  # Suppress specific warnings

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

    if data_file_name == 'clean1.data':
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
    
    elif data_file_name == 'vehicle.dat': 
        columns_name = ['COMPACTNESS', 'CIRCULARITY', 'DISTANCE CIRCULARITY', 'RADIUS RATIO', 'PR.AXIS ASPECT RATIO', 
                        'MAX.LENGTH ASPECT RATIO', 'SCATTER RATIO', 'ELONGATEDNESS', 'PR.AXIS RECTANGULARITY', 'MAX.LENGTH RECTANGULARITY', 
                        'SCALED VARIANCE ALONG MAJOR AXIS','SCALED VARIANCE ALONG MINOR AXIS', 'SCALED RADIUS OF GYRATION', 'SKEWNESS ABOUT MAJOR AXIS', 
                        'SKEWNESS ABOUT MINOR AXIS', 'KURTOSIS ABOUT MINOR AXIS', 'KURTOSIS ABOUT MAJOR AXIS', 'HOLLOWS RATIO', 'target']
        
        # Reading Column names from file
        with open(data_file,'r') as data_: 
            data = data_.readlines()
            dataset = pd.DataFrame(columns=columns_name)
            for line in data:
                x = line.split()
                dataset.loc[len(dataset)] = x
                
        dataset['class'] = [0 if row == 'opel' else 1 if row == 'saab' else 2 if row == 'van' else 3 for row in dataset['target']]
        dataset = dataset.drop(columns=['target'])
        return dataset

def data_normaliz(dataset):
    """
        Normalize the dataset
    """
    # Normalize the dataset
    scaler = MinMaxScaler()
    dataset_normlized = scaler.fit_transform(dataset)
    return dataset_normlized

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

    # Normalize the dataset
    X_normalized = data_normaliz(X)


    clf = LogisticRegression(max_iter=1000)  # Estimator
    cv = StratifiedKFold(5)

    rfecv = RFECV(
        estimator=clf,
        step=1,
        cv=cv,
        scoring="accuracy",
        #min_features_to_select=min_features_to_select,
        n_jobs=2,
    )
    rfecv.fit(X_normalized, y)
    #predictions = rfecv.predict(X_normalized)  # Predictions of the best subset of features

    #correct_predictions = sum(predictions == y)  # Number of correct predictions
    #error_rate = 1 - correct_predictions / len(y)  # Error rate

    selected_features = X.columns[rfecv.support_]  # Names of selected features
    #selected_feature_ratio = len(selected_features) / len(X.columns)  # Ratio of selected features

    # Get the accuracy of the best subset of features
    #accuracy = accuracy_score(y, rfecv.predict(X))
    
    return selected_features

def fitness_func(individual, dataset, target, seed_val=42):
    
    all_feature = dataset.shape[1]

    selected_features = [index for index in range(len(individual)) if individual[index] == 1]
    if len(selected_features) == 0:
        return 0
    
    X_selected = dataset.iloc[:, selected_features]
    
    # Normalizae the dataset
    X_selected_normalized = data_normaliz(X_selected)

    X_train, X_test, y_train, y_test = train_test_split(X_selected_normalized, target, test_size=0.3, random_state=seed_val)
    
    clf = RandomForestClassifier(n_estimators=50, random_state=seed_val)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_selected_normalized)
    
    # Objective 1: Error rate
    correct_predictions = sum(predictions != target)  # Number of correct predictions
    erro_rate = 1 - correct_predictions / len(target)  # Error rate
    
    # Objective 2: Feature selected ratio
    feature_selected_ratio = len(selected_features) / all_feature
    #accuracy = accuracy_score(y_test, predictions) # Accuracy
    
    return round(erro_rate, 4), round(feature_selected_ratio,4)

def get_individual_rank(individual, fronts):
    """
    Get the rank of a particular individual from the fronts.
    
    Parameters:
    individual: The individual whose rank is to be found.
    fronts: List of fronts where each front is a list of individuals.
    
    Returns:
    The rank (index) of the front where the individual is found, or None if not found.
    """
    for rank, front in enumerate(fronts):
        if individual in front:
            return rank
    return None

# Initialize population
def initialize_population(size, n_features, seed_val):
    np.random.seed(seed_val)
    return [np.random.randint(0, 2, n_features).tolist() for _ in range(size)]

# Evaluate individual
def evaluate_individual(individual, dataset):

    # Divide dataset into features and target
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Select features based on individual
    selected_features = [i for i, bit in enumerate(individual) if bit == 1]
    if len(selected_features) == 0:
        return 1.0, len(individual)  # Penalize individuals that select no features
    
    clf = KNeighborsClassifier()
    clf.fit(X_train[:, selected_features], y_train)
    predictions = clf.predict(X_test[:, selected_features])
    accuracy = accuracy_score(y_test, predictions)
    error = 1 - accuracy

    return error, len(selected_features)

# Non-dominated sorting
def non_dominated_sorting(population, fitnesses):
    # Hnadle case when population is empty
    if not population:
        return []
    
    # Initialization
    S = [[] for _ in range(len(population))]
    front = [[]]    # A list of lists of individuals in each front
    n = [0] * len(population)    # A list where n[p] is the number of solutions dominating p
    rank = [0] * len(population)  # A list where rank[p] is the rank (or front number) of individual p

    # Dominance Comparison
    for p in range(len(population)): # Loop for each individual in population
        for q in range(len(population)):
            if dominates(fitnesses[p], fitnesses[q]): # Check if p dominates q
                S[p].append(q)
            elif dominates(fitnesses[q], fitnesses[p]): # Check if q dominates p
                n[p] += 1 
        if n[p] == 0:  # if True, then p belongs to the first front
            rank[p] = 0 
            front[0].append(p) 

    # Constructing the subsequent fronts
    i = 0
    while front[i]: 
        Q = []  # A list to store the individuals for the next front
        for p in front[i]: # Loop for each individual in the current front
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0: # if True, then q belongs to the next front
                    rank[q] = i + 1 
                    Q.append(q)
        i += 1
        front.append(Q)

    del front[-1]  # Remove the last empty front
    return front

# Domination check
def dominates(fitness1, fitness2): 
    return all(x <= y for x, y in zip(fitness1, fitness2)) and any(x < y for x, y in zip(fitness1, fitness2))

# Crowding distance
def crowding_distance(fitnesses, front):
    distance = [0] * len(front)
    for i in range(len(fitnesses[0])):
        sorted_front = sorted(range(len(front)), key=lambda x: fitnesses[front[x]][i])

        if len(sorted_front) < 2: continue

        # Infinite distance for boundary individuals
        distance[sorted_front[0]] = distance[sorted_front[-1]] = float('inf')  # sorted_front[0]: This refers to the first element in the sorted front

        for j in range(1, len(sorted_front) - 1):
            distance[sorted_front[j]] += (fitnesses[sorted_front[j + 1]][i] - fitnesses[sorted_front[j - 1]][i]) / (max(fitnesses, key=lambda x: x[i])[i] - min(fitnesses, key=lambda x: x[i])[i])
    return distance

# Tournament selection
def tournament_selection(population, fronts, distance_populations, k=2):
    selected_parents = []
    while len(selected_parents) < k:
        select1, select2  = random.sample(range(len(population)), k)
        select1_rank = get_individual_rank(select1,fronts)  # Get the rank of the (select1) individual
        select2_rank = get_individual_rank(select2,fronts)  # Get the rank of the (select2) individual

        # First: Compare rank of the selected individuals.
        if select1_rank < select2_rank:  # If select1 has lower rank than select2, then select1 is better
            selected_parents.append(select1)
        elif select1_rank > select2_rank: # If select2 has lower rank than select1, then select2 is better
            selected_parents.append(select2)

        # Second: Compare Crowding Distance, If the two selected are in the same front, compare their crowding distances
        elif select1_rank == select2_rank:
            if distance_populations[select1] > distance_populations[select2]:
                selected_parents.append(select1)
            else:
                selected_parents.append(select2)

    return selected_parents[0], selected_parents[1]

# Crossover
def crossover(parent1, parent2, CROSSOVER_RATE):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2
        return child1, child2
    return parent1, parent2

# Mutation
def mutation(individual, MUTATION_RATE):
    if random.random() < MUTATION_RATE:
        index = random.randint(0, len(individual) - 1)
        individual[index] = 1 - individual[index]
    return individual

# Assuming fitness_func returns a list of objective values
def calculate_hypervolume(front_fitnesses):
    """
    Calculate the hyper-volume for a given front.
    
    Parameters:
    front: List of individuals in the front.
    reference_point: Reference point for hyper-volume calculation.
    
    Returns:
    Hyper-volume value.
    """
    # Normalize the fitness values
    scaler = MinMaxScaler()
    front_fitnesses_normalized = scaler.fit_transform(front_fitnesses)

    # Create the reference point (should be worse than any point in the front)
    reference_point = [1.1] * len(front_fitnesses[0])

    # Calculate the hyper-volume
    hv = HV(reference_point)
    hypervolume_value = hv.do(front_fitnesses_normalized)
    return hypervolume_value

# Function to plot Pareto front fitness values
def plot_pareto_front_fitness(pareto_front_fitness, run_no):
    """
    Plot the Pareto front fitness values.
    
    Parameters:
    pareto_front_fitness: List of fitness values for the Pareto front.
    """
    pareto_front_fitness = np.array(pareto_front_fitness)
    
    # two objectives, create a 2D scatter plot
    plt.scatter(pareto_front_fitness[:, 0], pareto_front_fitness[:, 1], color='b', marker='o')
    plt.title(f'Pareto Front Fitness Values for run ({run_no})')
    plt.xlabel('Obj1: Error Rate')
    plt.ylabel('Obj2: Feature Selected Ratio')

    plt.grid(True)
    plt.show()

def create_table(hyper_volume, mean_, std_):
    """
    Create a dataset with two columns from two input lists.

    Returns:
        pandas.DataFrame: A pandas DataFrame containing the two columns.
    """

    # First column
    first_column = ['Run 1','Run 2','Run 3']

    # Create a dictionary with the two lists as values
    data = {'': first_column, 'Hyper Volume': hyper_volume}

    # Create a pandas DataFrame from the dictionary
    data_table = pd.DataFrame(data)

    # Create a new DataFrame with the mean and concatenate it with (data_table)
    mean_row = pd.DataFrame({'': ['Mean'], 'Hyper Volume': [mean_]})
    data_table = pd.concat([data_table, mean_row], ignore_index=True)

    # Create a new DataFrame with the stander deviation and concatenate it with (data_table)
    std_row = pd.DataFrame({'': ['STD'], 'Hyper Volume': [std_] })
    data_table = pd.concat([data_table, std_row], ignore_index=True)

    return data_table

def main():
    # Parameters
    POPULATION_SIZE = 50
    GENERATIONS = 5
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.9
    RUNs = 3 
    SEED_VAL = [20, 40, 60]
    TURNAMENT_K = 2
    
    data_file =  ['vehicle.dat','clean1.data']    # wbcd.data  , sonar.data
    column_file = ['clean1.names']  # wbcd.names , sonar.names
    
    for idx, datafile_ in enumerate(data_file):
        hyper_volumes = []
        print('-----------------------------------------------------------------')
        print('Datafile: ', datafile_)
        # Load dataset 
        dataset = load_data(column_file[0], datafile_)
        Feature = dataset.iloc[:, :-2]
        Target = dataset.iloc[:, -1]

        for run in range(RUNs):
            print('Run:', run)
            fs = WrapperGA(Feature, Target)
            fs_datset = dataset[:][list(fs.array)]
            # Evolutionary algorithm
            population = initialize_population(POPULATION_SIZE, fs_datset.shape[1], SEED_VAL[run])
            
            for _ in range(GENERATIONS):
                print(f'\tGeneration {_} is running ...')
                # Fitness evaluation
                #fitnesses = [evaluate_individual(individual, ) for individual in population]
                fitnesses = [fitness_func(individual, fs_datset, Target.values, SEED_VAL[run]) for individual in population]

                # NON-DOMINANCE SORTING 
                fronts = non_dominated_sorting(population, fitnesses)

                # CROWDING DISTANCE: calcualte crowding distance. then sort individual based on distance for each front
                distance_populations = []
                for front in fronts:
                    distances = crowding_distance(fitnesses, front)
                    sorted_front = sorted(front, key=lambda x: distances[x] if x in distances else float('inf'))
                    distance_populations.extend(sorted_front)
                
                # Apply Elisim: Keep 

                # Apply Genetic Operation
                offspring = []
                while len(offspring) < POPULATION_SIZE:
                    parent1_idx, parent2_idx = tournament_selection(population, fronts, distance_populations,TURNAMENT_K)
                    child1, child2 = crossover(population[parent1_idx], population[parent2_idx], CROSSOVER_RATE)
                    child1 = mutation(child1, MUTATION_RATE)
                    child2 = mutation(child2, MUTATION_RATE)
                    offspring.append(child1)
                    offspring.append(child2)
                

                # Combine new population(offspring) with old population
                combine_population = population + offspring
                combine_population_fitness = fitnesses + [fitness_func(individual, fs_datset, Target.values, SEED_VAL[run]) for individual in offspring]
                
                # Combine fronts and distance
                combine_fronts = non_dominated_sorting(combine_population, combine_population_fitness)
                combine_distance = []
                for front in combine_fronts:
                    distances = crowding_distance(combine_population_fitness, front)
                    sorted_front = sorted(front, key=lambda x: distances[x] if x in distances else float('inf'))
                    combine_distance.extend(sorted_front)


                # for loop for each front. and add individual to next generation population up to reach POPULATION_SIZE
                new_population = []
                for front in combine_fronts:
                    # check if population size is less than POPULATION_SIZE, then add all individual of front to next generation (new_population)
                    if len(front) + len(new_population) <= POPULATION_SIZE:
                        # Add all individual of front to new_population
                        [new_population.append(combine_population[ind]) for ind in front]
                    else:
                        # Check how many individual can be added to next generation
                        remaining = POPULATION_SIZE - len(new_population)
                        if remaining == 0:  # If no individual can be added to next generation, then go to next generation
                            population = new_population
                            break
                        # Sort front based on crowding distance, then add remaining individual to next generation
                        front_distance_sort = sorted(front, key=lambda x: combine_distance[x], reverse=True)
                        [new_population.append(combine_population[ind]) for ind in front_distance_sort[:remaining]]
                        
                        population = new_population
                        break

                # Apply Stoping Crieteria. 

            # HYPER-VOLUME CALCULATION
            # Clculate non-dominance sorting for final population
            new_population_fitness = [fitness_func(ind, fs_datset, Target.values, SEED_VAL[run]) for ind in population]
            new_population_fronts = non_dominated_sorting(population, new_population_fitness)
            
            # Extract the first front (front 0 and its fitness from the new_population
            #pareto_front = [population[ind] for ind in new_population_fronts[0]]
            pareto_front_fitness = [new_population_fitness[ind] for ind in new_population_fronts[0]]
            print('Pareto Front (index of population):', new_population_fronts[0])

            # HYPER-VOLUME: Calculate the hyper-volume for front 0, then plot the hypervolume over iterations
            hyper_vol = calculate_hypervolume(pareto_front_fitness)
            hyper_volumes.append(hyper_vol)
            print('Hyper Volume:', round(hyper_vol, 4))

            # Plot the Pareto front fitness values
            plot_pareto_front_fitness(pareto_front_fitness, run)
            print('-----------------------------------------------------------------')

        # Calculate the mean and standard deviation of the hyper-volume values
        mean_hyper_volume = np.mean(hyper_volumes)
        std_hyper_volume = np.std(hyper_volumes)
        
        # Create a table with the hyper-volume values, mean, and standard deviation
        data_table = create_table(hyper_volumes, mean_hyper_volume, std_hyper_volume)
        print(data_table)


if __name__ == "__main__":
    main()