#get the node definitions from ods

import pyodbc
import time
#measure elapsed time - starting point
start = time.time()
connection = pyodbc.connect("DRIVER={PostgreSQL Unicode(x64)};SERVER=#####;DATABASE=plethora;UID=####;PWD=####;sslmode=allow")
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
d = 'c://users//jswortz//desktop//txl2'
data = []
qa = []
files = os.listdir(d)
for f in files[-10:]:
    tar= tarfile.open(d + '/' + f, "r:gz")
    for x in tar.getmembers(): #iterating over fist few tar files
        f = tar.extractfile(x)
        line = f.read()
        line = line.strip()
        columns = line.split('|')
        end = time.time()
        print end - start
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
    
    
NNGen(train=nnet_data,valid=nnet_data_test, n_inputs=n_x,iter_min=200,iter_max=500, n_nodes_min=2,n_nodes_max=8,learn_min=.1,
learn_max=.8,m_min=.1,m_max=.8,CXPB=.5, MUTPB=.2, NGEN=10, tot_pop=50)    
    
    
#establish nn framework
n = NN(n_x, 4, 1)
n.train(nnet_data,iterations=2000,N=.7)
testData = n.test(nnet_data_test)
#calc rmse
sse = 0
n = 0
for i in testData:
    sse += (i[0][0]-i[1][0])**2
    n += 1
mse = sse/n
rmse = (mse)**.5

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



if __name__ == '__main__':
    demo()
