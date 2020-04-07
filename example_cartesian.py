import geneticalgorithm

##How To setup your our GA for Cartesian
# Basically your need to overwrite the eval function, the function that calculates the value for the individual
class MyGACart(GeneticAlgorithm):
    class Individual(GeneticAlgorithm.Individual):
        def __cartesian_eval(self, point):
            res = 0
            for i in range(len(point)):
                res = res + (point[i] ** 4 - 16 * point[i] ** 2 + 5 * point[i])
            return res/2

# Prepare kw_args
my_kw_args = {}
my_kw_args[0] = {"ga_type":'Cartesian', "boundary":[(-4,4),(-4,4)], "population_size":100, "mutation_prob":0.01, "cross_prob":0.8, "init_seed":None,
           "max_it":100, "min_error":1E-8, "min_error_length":10, "plot":True, "verbose":False}
# Define runs
my_runs = 5
# Define your GA class
my_ga_class = 'MyGACart'
# Instanciate GAManager
MyGAM = GAManager(run=my_runs, ga_class=my_ga_class, kw_args=my_kw_args)
# Run GAManager
print("GA Manager run\n" + 14*"-" + "\n")
MyGAM.run()

# Or just run a single run with the GA instance
print("\nSignle run with the GA Instance\n" + 31*"-" + "\n")
MyGAInst = MyGACart(**my_kw_args[0])
MyGAInst.fit()
MyGAInst.get_results()
