import geneticalgorithm

##How To setup your our GA for Permutation with Euclidean Distance
# Basically your need to overwrite the eval function, the function that calculates the distance of two elements in the set
# i.e. distance of two cities. You can calculate Euclidean distance or any kind of distance you want, you can use your own
# customized function to do that
class MyGAPerm(GeneticAlgorithm):
    class Individual(GeneticAlgorithm.Individual):
        def __permutation_eval(self, element1, element2):
            xDis = abs(element1[0] - element2[0])
            yDis = abs(element1[1] - element1[1])
            return np.sqrt((xDis ** 2) + (yDis ** 2))

# Generate Random cartesian points, simulating city coordenates
points_list = []
for i in range(20):
    points_list.append((random.uniform(0,200),random.uniform(0,200)))

# Prepare kwargs
my_kw_args = {}
my_kw_args[0] = {"ga_type":'Permutation', "points":points_list, "population_size":100, "mutation_prob":0.01, "cross_prob":0.8, "init_seed":7,
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
