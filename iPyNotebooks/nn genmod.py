def NNGen(train,valid, n_inputs,iter_min=500,iter_max=1000, n_nodes_min=2,n_nodes_max=4,learn_min=.1,
learn_max=.9,m_min=.1,m_max=.9,CXPB=.5, MUTPB=.2, NGEN=40, tot_pop=300):
    
        # create a network with two input, two hidden, and one output nodes
    def NNError(y ,bb, learn, m):
        n = NN(n_inputs, y, 1)
        # train it with some patterns
        n.train(train,iterations=bb, N=learn, M=m)
        # test it
        x = n.test(valid)
        y = n.test(train)
        a = 0.0
        tp = 0.0
        fp = 0.0
        tn = 0.0
        fn = 0.0
        av = 0.0
        tpv = 0.0
        fpv = 0.0
        tnv = 0.0
        fnv = 0.0
        for i in y:
            j=(i[0][0]-i[1][0])**2
            a+=j
            if i[1][0] > .5:
                if i[0][0] == 1:
                    tp=+1
                else:
                    if i[0][0] == 0:
                        fp=+1
            if i[1][0] <= .5:
                if i[0][0] ==0:
                    tn=+1
                else:
                    if i[0][0] ==1:
                        fn =+1
        for i in x:
            j=(i[0][0]-i[1][0])**2
            av+=j
            if i[0][0] > .5:
                if i[1][0] == 1:
                    tpv=+1
                else:
                    if i[1][0] == 0:
                        fpv=+1
            if i[0][0] <= .5:
                if i[1][0] ==0:
                    tnv=+1
                else:
                    if i[1][0] ==1:
                        fnv =+1
        recallv = tpv / (tpv + fnv)
        specifv = tnv / (tnv + fpv)
        precisv = tpv / (tpv + fpv)
        accv = (tpv + tnv) / (tpv + fpv + tnv + fnv)
        f1v = 2*tpv/(2*tpv + fpv + fnv)                     
        rmsev = av**.5
        recall = tp / (tp + fn)
        specif = tn / (tn + fp)
        precis = tp / (tp + fp)
        acc = (tp + tn) / (tp + fp + tn + fn)
        f1 = 2*tp/(2*tp + fp + fn)                     
        rmse = a**.5
        #return rmse, recall, specif, precis, acc, f1, rmsev, recallv, specifv, precisv, accv, f1v
        delta = abs(rmse - rmsev)
        return acc,delta
    
    #NNError(bb=100,learn=.5,m=.1)
    
    def objective(x):
        a=x[0]
        b=x[1]
        c=x[2]
        d=x[3]
        return NNError(y = d, bb=a,learn=b,m=c)
    start = time.time()
 
    
    
    #    This file is part of DEAP.
    #
    #    DEAP is free software: you can redistribute it and/or modify
    #    it under the terms of the GNU Lesser General Public License as
    #    published by the Free Software Foundation, either version 3 of
    #    the License, or (at your option) any later version.
    #
    #    DEAP is distributed in the hope that it will be useful,
    #    but WITHOUT ANY WARRANTY; without even the implied warranty of
    #    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    #    GNU Lesser General Public License for more details.
    #
    #    You should have received a copy of the GNU Lesser General Public
    #    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.
    
    
    #    example which maximizes the sum of a list of integers
    #    each of which can be 0 or 1
    
    import random
    
    from deap import base
    from deap import creator
    from deap import tools
    
    creator.create("Fitness", base.Fitness, weights=(-1.0,-1.0))
    creator.create("Individual", list, fitness=creator.Fitness)
    
    toolbox = base.Toolbox()
    
    # Attribute generator: define 'attr_bool' to be an attribute ('gene')
    #                      which corresponds to integers sampled uniformly
    #                      from the range [0,1] (i.e. 0 or 1 with equal
    #                      probability)
    toolbox.register("iterations", random.randint, iter_min, iter_max)
    toolbox.register("learn",  random.uniform, learn_min, learn_max)
    toolbox.register("rate", random.uniform, m_min, m_max)
    toolbox.register("nodes", random.randint, n_nodes_min, n_nodes_max)
    
    # Structure initializers: define 'individual' to be an individual
    #                         consisting of 100 'attr_bool' elements ('genes')
    toolbox.register("individual",tools.initCycle, creator.Individual, 
        (toolbox.iterations, toolbox.learn, toolbox.rate, toolbox.nodes), n=1)
    
    # define the population to be a list of 'individual's
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # the goal ('fitness') function to be maximized
    def evalOneMax(individual):
        return sum(individual),
    
    #----------
    # Operator registration
    #----------
    # register the goal / fitness function
    toolbox.register("evaluate", objective)
    
    # register the crossover operator
    toolbox.register("mate", tools.cxTwoPoint)
    
    # register a mutation operator with a probability to
    # flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    
    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    #----------
    
    def NNOpt():
        random.seed(64)
    
        # create an initial population of 300 individuals (where
        # each individual is a list of integers)
        pop = toolbox.population(n=tot_pop)
    
        # CXPB  is the probability with which two individuals
        #       are crossed
        #
        # MUTPB is the probability for mutating an individual
        #
        # NGEN  is the number of generations for which the
        #       evolution runs
        #CXPB, MUTPB, NGEN = 0.5, 0.2, 40
        
        print("Start of evolution")
        
        # Evaluate the entire population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        print("  Evaluated %i individuals" % len(pop))
        
        # Begin the evolution
        for g in range(NGEN):
            print("-- Generation %i --" % g)
            
            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))
        
            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
    
                # cross two individuals with probability CXPB
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
    
                    # fitness values of the children
                    # must be recalculated later
                    del child1.fitness.values
                    del child2.fitness.values
    
            for mutant in offspring:
    
                # mutate an individual with probability MUTPB
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
        
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            print("  Evaluated %i individuals" % len(invalid_ind))
            
            # The population is entirely replaced by the offspring
            pop[:] = offspring
            
            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]
            
            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5
            
            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)
            
            fits = [ind.fitness.values[1] for ind in pop]
            
            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5
            
            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)
        
        print("-- End of (successful) evolution --")
        
        best_ind = tools.selBest(pop, 1)[0]
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
        end = time.time()
        print end - start
    return NNOpt()




