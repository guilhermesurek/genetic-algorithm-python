import geneticalgorithm

# Run GA Manager - Permutation

# Prepare kwargs
kw_args_0 = {"ga_type":'Permutation', "points":points_list, "population_size":100, "mutation_prob":0.01, "cross_prob":0.8, "init_seed":7,
           "max_it":200, "min_error":1E-8, "min_error_length":10, "plot":True, "verbose":False}
kw_args_1 = {"ga_type":'Permutation', "points":points_list, "population_size":100, "mutation_prob":0.02, "cross_prob":0.8, "init_seed":7,
           "max_it":200, "min_error":1E-8, "min_error_length":10, "plot":True, "verbose":False}
kw_args_2 = {"ga_type":'Permutation', "points":points_list, "population_size":100, "mutation_prob":0.05, "cross_prob":0.8, "init_seed":7,
           "max_it":200, "min_error":1E-8, "min_error_length":10, "plot":True, "verbose":False}
kw_args_3 = {"ga_type":'Permutation', "points":points_list, "population_size":100, "mutation_prob":0.1, "cross_prob":0.8, "init_seed":7,
           "max_it":200, "min_error":1E-8, "min_error_length":10, "plot":True, "verbose":False}
kw_args_4 = {"ga_type":'Permutation', "points":points_list, "population_size":100, "mutation_prob":0.01, "cross_prob":0.7, "init_seed":7,
           "max_it":200, "min_error":1E-8, "min_error_length":10, "plot":True, "verbose":False}

# Set kwargs
kw_args = {}
kw_args[0] = kw_args_0
kw_args[1] = kw_args_1
kw_args[2] = kw_args_2
kw_args[3] = kw_args_3
kw_args[4] = kw_args_4

RUNS = 5
GAM = GAManager(RUNS, "GeneticAlgorithm", kw_args)
GAM.run()
