
# coding: utf-8

# In[1]:

#bpnn


# Back-Propagation Neural Networks
# 
# Written in Python.  See http://www.python.org/
# Placed in the public domain.
# Neil Schemenauer <nas@arctrix.com>

import math
import random
import string

random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        
        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum   
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error

    def test(self, patterns):
        pp = []
        for p in patterns:
            qq = []
            #print(p[0], '->', self.update(p[0]))
            qq.append(p[1])
            qq.append(self.update(p[0]))
            pp.append(qq)
        return(pp)

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0:
                print('error %-.5f' % error)


def demo():
    # Teach network XOR function
    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]

    # create a network with two input, two hidden, and one output nodes
    n = NN(2, 2, 1)
    # train it with some patterns
    n.train(pat)
    # test it
    n.test(pat)


# In[3]:

def fitStats(data):  
   a = 0.0
   tp = 0.0
   fp = 0.0
   tn = 0.0
   fn = 0.0
   av = 0.0
   for i in data:
       j=(i[0]-i[1])**2
       a+=j
       if i[1] > .5:
           if i[0] == 1:
               tp+=1
           else:
               if i[0] == 0:
                   fp+=1
       if i[1] <= .5:
           if i[0] ==0:
               tn+=1
           else:
               if i[0] ==1:
                   fn+=1
   try:
       recall = tp / (tp + fn)
       specif = tn / (tn + fp)
       precis = tp / (tp + fp)
       acc = (tp + tn) / (tp + fp + tn + fn)
       f1 = 2*tp/(2*tp + fp + fn) 
   except:
       recall, specif, precis, acc, f1 = 0, 0, 0, 0, 0
   rmse = a**.5
   return rmse, recall, specif, precis, acc, f1


# In[ ]:

#get the node definitions from ods

import pyodbc
import time
#measure elapsed time - starting point
start = time.time()
connection = pyodbc.connect("DRIVER={PostgreSQL Unicode(x64)};SERVER=odsvip01;DATABASE=plethora;UID=####;PWD=####;sslmode=allow")
cursor = connection.cursor()
cursor.execute("select distinct node_name, case when event_type_code = 'SS' then 1 end as ss, " "case when event_type_code in ('INFD','Messaging','Service check - INFD') then 1 end as INFD from txl.Client_billing_tier where event_type_code in ('INFD','Messaging','Service check - INFD','SS')")
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
for f in files[-100:]:
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
    if (    #(row['sales_flag'] == 1) or \
    (row['ss_flag'] == 1 and row['networkxfer'] != 1) or (row['inf_flag'] == 1 and row['networkxfer'] != 1)):
        return 1
df['tier1'] = df.apply(func, axis=1)
cols = list(df.columns)
r = ['id','ss_flag','networkxfer','inf_flag']
for item in r:
    cols.remove(item)

df2 = pd.get_dummies(df,columns=cols)

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
    


# In[18]:




# In[ ]:

import random
    
from deap import base
from deap import creator
from deap import tools

creator.create("Fitness", base.Fitness, weights=(1.0,-1.0))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()

toolbox.register("nodes", random.randint,5,30)
toolbox.register("N", random.uniform, 0,1)
toolbox.register("M", random.uniform, 0,1)


toolbox.register("individual",tools.initCycle, creator.Individual, 
    (toolbox.nodes, toolbox.N, toolbox.M), n=1)

# define the population to be a list of 'individual's
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
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
Yavg = sum(Y)/len(Y)
    
    

def objective(ind):
    
    n = NN(n_x, ind[0], 1)
    # train it with some patterns
    n.train(nnet_data,iterations=5000,N=ind[1],M=ind[2])
    # test it
    y = n.test(nnet_data_test)
    x = n.test(nnet_data)
    yq = []
    xq = []
    for i in y:
        if i[1] > Yavg:
            yq.append(1)
        else:
            yq.append(0)
    for i in x:
        if i[1] > Yavg:
            yq.append(1)
        else:
            yq.append(0)

    valset = zip(Yv,yq)
    statsV = fitStats(valset)
    trainset = zip(Y,xq)
    statsT = fitStats(trainset)
    overfit = abs(statsT[4]-statsV[4])
    return statsV[4], overfit
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
toolbox.register("select", tools.selTournament, tournsize=4)

#----------

def NNOpt():
    random.seed(64)

    pop = toolbox.population(n=100)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    #
    # NGEN  is the number of generations for which the
    #       evolution runs
    CXPB, MUTPB, NGEN = 0.5, 0.2, 15

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

NNOpt()






# In[19]:

objective(toolbox.individual())

