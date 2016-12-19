import numpy as np
import neurolab as nl
import pickle
df2 = pickle.load(open("ComcastData2","rb"))
cutoff = .05
cols = list(df2.columns)
for col in cols:
    if df2[col].mean() < cutoff:
        cols.remove(col)
for c in cols:
    if c.endswith('_OFF'):
        cols.remove(str(c))
r = ['id','ss_flag','networkxfer','inf_flag']
#remove dependent var from list
for item in r:
    cols.remove(item)
df2 = df2[cols]
q = df2.corr()
for c in q
cols.remove('tier1_1.0')
import numpy as np
#partition training and test
msk = np.random.rand(len(df2)) < 0.8
train = df2[msk]
test = df2[~msk]



shape = train[cols].shape
n_x = shape[1]
n_n = shape[0]

#hidden node upper bound calculation
up_bound1 = n_n / (n_x +1) / 5
up_bound2 = n_n / (n_x +1) / 10

inputs = train[cols].values
test_data = test[cols].values
outputs = train['tier1_1.0'].values.tolist()
z = zip(inputs,outputs)
outputs_test = test['tier1_1.0'].values.tolist()
z_test = zip(test_data,outputs_test)
nnet_data = []
for item in z:
    b=[]
    b.append(item[1])
    r = item[0].tolist()
    v = []
    v.append(r)
    v.append(b)
    nnet_data.append(v)
nnet_data_test = []
for item in z_test:
    b=[]
    b.append(item[1])
    r = item[0].tolist()
    v = []
    v.append(r)
    v.append(b)
    nnet_data_test.append(v)


X = []
Y = []
for i in nnet_data:
    X.append(i[0])
    Y.append(i[1][0])
Xv = []
Yv = []
for i in nnet_data_test:
    Xv.append(i[0])
    Yv.append(i[1][0])
    
train = np.array(X)
trainY = np.array(Y)
trainY = trainY.reshape(len(trainY),1)
valid = np.array(Xv)
validY = np.array(Yv)
validY = validY.reshape(len(validY),1)
import itertools

inputs = list(itertools.repeat([0,1],n_x))
net = nl.net.newff(inputs, [15,10,5, 1])
net.ci = n_x
net.co = 1
len(net.layers)
net.train(train,trainY,goal=.01,show=500,epochs=500)
out = net.sim(train)
size = len(train)
line = np.random.randint(2,size=n_x*1)
line = line.reshape(1,n_x)
predY = net.sim(valid)
classPred = []
avg = trainY.mean()
for i in predY:
    if i[0] >avg:
        classPred.append(1)
    else:
        classPred.append(0)
   
from sklearn.metrics import confusion_matrix
confusion_matrix(classPred,validY)
import pickle
pickle.dump(net,open("ComcastNN","wb"))

import pylab as pl

pl.subplot(211)
pl.plot(error)
pl.xlabel('Epoch number')
pl.ylabel('error (default SSE)')

x1 = np.linspace(0,1,1*n_x)
x2 = x1.reshape(1,n_x)
y2 = net.sim(x2)
y3 = out.reshape(size)

#pl.subplot(212)
#pl.plot(x2, y2, '-',x , y, '.', x, y3, 'p')
#pl.legend(['train target', 'net output'])
pl.show()




import random
from deap import base
from deap import creator
from deap import tools
    



def objective2(inputs):
    inputs = np.array(inputs)
    inputs = inputs.reshape(1,n_x)
    p = net.sim(inputs)
    return p[0],


creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()

toolbox.register("bins", random.randint,0,1)



toolbox.register("individual",tools.initRepeat, creator.Individual, 
   toolbox.bins, n=n_x)

# define the population to be a list of 'individual's
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


#----------
# Operator registration
#----------
# register the goal / fitness function
toolbox.register("evaluate", objective2)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=4)

#----------

def NNOpt():
    random.seed(64)

    pop = toolbox.population(n=1000)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    #
    # NGEN  is the number of generations for which the
    #       evolution runs
    CXPB, MUTPB, NGEN = 0.5, 0.2, 150

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


    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    end = time.time()
    print end - start
    return best_ind

best = NNOpt()