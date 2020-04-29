import geneticalgorithm

##How To setup your our GA for Permutation with weighted matrix
# Basically your need to overwrite the eval function, the function that calculates the distance of two elements in the set
# i.e. distance of two cities. You can calculate Euclidean distance or any kind of distance you want, you can use your own
# customized function to do that

# Define weighted matrix
dist_mat = [[0, 2, 8, 4, 6, 17, 42], 
            [2, 0, 14, 3, 9, 15, 38], 
            [8, 14, 0, 5, 4, 24, 17], 
            [4, 3, 5, 0, 20, 57, 32], 
            [6, 9, 4, 20, 0, 29, 98],
            [17, 15, 24, 57, 29, 0, 26],
            [42, 38, 17, 32, 98, 26, 0]]

class MyGAPerm(GeneticAlgorithm):
    class Individual(GeneticAlgorithm.Individual):
        def __permutation_eval(self, element1, element2):
            return dist_mat[element1][element2]

# Generate Random cartesian points, simulating city coordenates
points_list = [0,1,2,3,4,5,6]        # Simulating each number is a city

# Prepare kwargs
my_kw_args = {}
my_kw_args[0] = {"ga_type":'Permutation', "points":points_list, "population_size":10, "mutation_prob":0.01, "cross_prob":0.8, "init_seed":None,
                     "max_it":200, "min_error":1E-8, "min_error_length":10, "plot":True, "verbose":False}

# Define runs
my_runs = 5
# Define your GA class
my_ga_class = 'MyGAPerm'
# Instanciate GAManager
MyGAM = GAManager(run=my_runs, ga_class=my_ga_class, kw_args=my_kw_args)
# Run GAManager
print("GA Manager run\n" + 14*"-" + "\n")
MyGAM.run()

# Or just run a single run with the GA instance
print("\nSignle run with the GA Instance\n" + 31*"-" + "\n")
MyGAInst = MyGAPerm(**my_kw_args[0])
MyGAInst.fit()
MyGAInst.get_results()
