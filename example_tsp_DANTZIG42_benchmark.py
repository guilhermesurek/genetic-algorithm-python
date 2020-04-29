import geneticalgorithm

## Download Weighted Matrix
target_url = 'https://people.sc.fsu.edu/~jburkardt/datasets/tsp/dantzig42_d.txt'

import urllib.request  # the lib that handles the url stuff
import pandas as pd
import numpy as np

raw = []
for line in urllib.request.urlopen(target_url):
    raw.append(line.decode("utf-8").split())

df_case_1 = pd.DataFrame(raw,index=None).astype(int)

##Setting up code to run case 1

# Setup Base Code
class GACase1(GeneticAlgorithm):
    class Individual(GeneticAlgorithm.Individual):
        def __permutation_eval(self, element1, element2):
            return df_case_1[element1][element2]

# Generate city list
points_list = [i for i in range(42)]

# Prepare kwargs
my_kw_args = {}
my_kw_args[0] = {"ga_type":'Permutation', "points":points_list, "population_size":100, "mutation_prob":0.005, "cross_prob":0.7, "init_seed":7,
                 "max_it":300, "min_error":1E-8, "min_error_length":15, "plot":True, "verbose":False}
my_kw_args[1] = {"ga_type":'Permutation', "points":points_list, "population_size":100, "mutation_prob":0.008, "cross_prob":0.7, "init_seed":7,
                 "max_it":300, "min_error":1E-8, "min_error_length":15, "plot":True, "verbose":False}
my_kw_args[2] = {"ga_type":'Permutation', "points":points_list, "population_size":100, "mutation_prob":0.01, "cross_prob":0.7, "init_seed":7,
                 "max_it":300, "min_error":1E-8, "min_error_length":15, "plot":True, "verbose":False}
my_kw_args[3] = {"ga_type":'Permutation', "points":points_list, "population_size":200, "mutation_prob":0.005, "cross_prob":0.7, "init_seed":7,
                 "max_it":300, "min_error":1E-8, "min_error_length":15, "plot":True, "verbose":False}
my_kw_args[4] = {"ga_type":'Permutation', "points":points_list, "population_size":100, "mutation_prob":0.005, "cross_prob":0.8, "init_seed":7,
                 "max_it":300, "min_error":1E-8, "min_error_length":15, "plot":True, "verbose":False}
my_kw_args[5] = {"ga_type":'Permutation', "points":points_list, "population_size":100, "mutation_prob":0.005, "cross_prob":0.6, "init_seed":7,
                 "max_it":300, "min_error":1E-8, "min_error_length":15, "plot":True, "verbose":False}

# Define runs
my_runs = 30
# Define your GA class
my_ga_class = 'GACase1'
# Instanciate GAManager
GAM_1 = GAManager(run=my_runs, ga_class=my_ga_class, kw_args=my_kw_args)
# Run GAManager
print("-"*8 + "\nCASE 1\n" + "GA Manager run\n" + 14*"-" + "\n")
GAM_1.run()
