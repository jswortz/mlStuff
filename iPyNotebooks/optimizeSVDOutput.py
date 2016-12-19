#get the node definitions from ods

import pyodbc
import time
#measure elapsed time - starting point
start = time.time()
connection = pyodbc.connect("DRIVER={PostgreSQL Unicode(x64)};SERVER=####;DATABASE=plethora;UID=####;PWD=####;sslmode=allow")
cursor = connection.cursor()
cursor.execute("select distinct node_name, case when event_type_code = 'SS' then 1 end as ss, " \
"case when event_type_code in ('INFD','Messaging','Service check - INFD') then 1 end as INFD from txl.Client_billing_tier where event_type_code in ('INFD','Messaging','Service check - INFD','SS')")
luptable = []
for row in cursor:
    luptable.append(row)
#get apns from client
cursor.execute("select distinct apn from ods.program_apn apn inner join ods.ods_client cli on (apn.client_identifier = cli.client_identifier) where cli.client_number = 526993")
apns = []
for row in cursor:
    row = str(row[0])
    apns.append(row)

connection.close()

#create SS flag items
ss_flag = []
for row in luptable:
    if row[1] == 1:
        ss_flag.append(row[0])
#create infd items
inf_flag = []
for row in luptable:
    if row[2] == 1:
        inf_flag.append(row[0])
        
#mapper portion, change file specs to stdin
import os
import tarfile
d = '//wcw30390/SASDATA/WIC/TXLRaw'
data = []
qa = []
files = os.listdir(d)
for f in files[-200:]:
    tar= tarfile.open(d + '/' + f, "r:gz")
    for x in tar.getmembers(): #iterating over fist few tar files
        f = tar.extractfile(x)
        line = f.read()
        line = line.strip()
        columns = line.split('|')
        #print(columns)
        try:
            if str(columns[5]) in apns:
                receipt = line.split('receive>')
                receipt[1].replace('|/>','')
                ids = columns[2] + columns[3] + columns[4]
                apis = {'id':ids}
                kvs = receipt[1].split('|')
                if columns[7] in ss_flag:
                    apis.update({'ss_flag':1})
                if columns[7] in inf_flag:
                    apis.update({'inf_flag':1})
                if columns[13].startswith('networkxfer'):
                    apis.update({'networkxfer':1})
                #if columns[1] == 'Inbound':
                    #qa.append(columns) ##checking for issues with last call record
                if columns[7] == 'prompt_AcctTypeNew':
                    apis.update({'sales_flag':1})
                for kz in kvs:
                    vals = kz.split('=')
                    k = vals[0].split(',')
                    if k[1] == 'toggle':
                        vals[1] = vals[1].replace('</','')
                        apis.update({vals[0]: vals[1]})
                data.append(apis.copy())                            
        except:
            pass
    tar.close()
end = time.time()
print end - start
##flatten the call records
kez = []
for item in data:
    key = item['id']
    if key not in kez:
        kez.append(key)
d2 = []
for k in kez:
    holder = {}
    for item in data:
        if item['id'] == k:
                holder.update(item)
    d2.append(holder)

#define tier 1

    
            
import pandas as pd

df = pd.DataFrame(d2)

def func(row):
    if (\
    #(row['sales_flag'] == 1) or \
    (row['ss_flag'] == 1 and row['networkxfer'] != 1) or (row['inf_flag'] == 1 and row['networkxfer'] != 1)):
        return 1
df['tier1'] = df.apply(func, axis=1)
cols = list(df.columns)
r = ['id','ss_flag','networkxfer','inf_flag']
for item in r:
    cols.remove(item)

df2 = pd.get_dummies(df,columns=cols)
import pickle
pickle.dump(df2, open("ComcastData2","wb")) 

#remove dependent var from list
cols = list(df2.columns)
for item in r:
    cols.remove(item)
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
        
import random

from deap import base
from deap import creator
from deap import tools
from sklearn import svm
    
best = [1.21,-1.71,.836,1,2]
def bestModel(ind):
    ker = ['rbf','poly','sigmoid','linear']
    k = ker[ind[3]]
    clf = svm.SVC(C=10**ind[0],gamma=10**ind[1],coef0=0,kernel=k, degree=ind[4],probability=True)
    clf.fit(X,Y)
    #clf.fit(X, Y) 
    #valset = zip(Yv,clf.predict(Xv))
    #statsV = fitStats(valset)
    #trainset = zip(Y,clf.predict(X))
   # statsT = fitStats(trainset)
    #overfit = abs(statsT[4]-statsV[4])
    return clf

bestMod = bestModel(best)

def objective2(inputs):
    p = bestMod.predict_proba(inputs)
    return p[0][1],


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

NNOpt()
df3 = pd.DataFrame(X)
mode = df3.mode()
ml = mode.values.tolist()
objective2(ml)