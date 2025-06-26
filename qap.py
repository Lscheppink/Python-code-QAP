import numpy as np
import sys
import time
from CalculateALU import *
import json

#import data
if len(sys.argv) <= 1:
    filename = 'bur26a.dat'
else:
    filename = sys.argv[1]

results = {'instance': filename,'relaxations': {}}

with open(filename) as f:
    lines = [line.strip() for line in f if line.strip()]
n = int(lines[0])                                   # The first element is the size of the instance
data = []
for line in lines[1:]:
    data.extend(map(int, line.split()))             # Puts all the numbers into 1 big list
full_matrix = np.array(data).reshape((2 * n, n))

F = full_matrix[:n]
D = full_matrix[n:]
c = np.zeros((n,n,n,n))
for i in range(n):
    for j in range(n):
        for k in range(n):
            for l in range(n):
                c[i,j,k,l] = F[i,k] * D[j,l]

#Precompute a, l and u
A = calculate_A(c)
L = calculate_L(c)
U = calculate_U(c)



#LL
model = Model('LL_LP_Relaxation')
model.params.method = 2
model.params.crossover = 0
model.params.SoftMemLimit = 16

x = model.addVars(n, n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='x')
y = model.addVars(n, n, n, n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='y')
model.update()

model.setObjective(
    sum(c[i,j,k,l] * y[i,j,k,l] for i in range(n) for j in range(n) for k in range(n) for l in range(n)),
    GRB.MINIMIZE)

for i in range(n):
    model.addConstr(sum(x[i,j] for j in range(n)) == 1)
for j in range(n):
    model.addConstr(sum(x[i,j] for i in range(n)) == 1)
model.addConstr(sum(y[i,j,k,l] for i in range(n) for j in range(n)
                                   for k in range(n) for l in range(n)) == n**2)
for i in range(n):
    for j in range(n):
        for k in range(n):
            for l in range(n):
                model.addConstr(x[i,j] + x[k,l] - 2 * y[i,j,k,l] >= 0)

model.update()
model.setParam('TimeLimit', 3600)
start_time = time.time()
model.optimize()
if model.status == GRB.OPTIMAL:
    end_time = time.time()
    runtime = round(end_time - start_time, 2)
    objective = round(model.objVal)
    results['relaxations']['LL'] = {'time': runtime, 'objective': objective}
elif model.status == GRB.TIME_LIMIT:
    end_time = time.time()
    runtime = round(end_time - start_time, 2)
    results['relaxations']['LL'] = {'time': runtime, 'status': 'time_limit'}
elif model.status == GRB.MEM_LIMIT:
    end_time = time.time()
    runtime = round(end_time - start_time, 2)
    results['relaxations']['LL'] = {'time': runtime, 'status': 'soft_memory_limit'}
else:
    results['relaxations']['LL'] = {'status': 'not_solved'}




# FYL
model = Model('FYL_LP_Relaxation')
model.params.method = 2
model.params.crossover = 0
model.params.SoftMemLimit = 16
x = model.addVars(n, n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='x')
y = model.addVars(n, n, n, n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='y')

model.update()

model.setObjective(
    sum(c[i,j,k,l] * y[i,j,k,l] for i in range(n) for j in range(n) for k in range(n) for l in range(n)),
    GRB.MINIMIZE)

for i in range(n):
    model.addConstr(sum(x[i,j] for j in range(n)) == 1)
for j in range(n):
    model.addConstr(sum(x[i,j] for i in range(n)) == 1)

for j in range(n):
    for k in range(n):
        for l in range(n):
            model.addConstr(sum(y[i,j,k,l] for i in range(n)) == x[k,l])

for i in range(n):
    for k in range(n):
        for l in range(n):
            model.addConstr(sum(y[i,j,k,l] for j in range(n)) == x[k,l])

for i in range(n):
    for j in range(n):
        for l in range(n):
            model.addConstr(sum(y[i,j,k,l] for k in range(n)) == x[i,j])

for i in range(n):
    for j in range(n):
        for k in range(n):
            model.addConstr(sum(y[i,j,k,l] for l in range(n)) == x[i,j])

for i in range(n):
    for j in range(n):
        model.addConstr(y[i,j,i,j] == x[i,j])

model.update()
model.setParam('TimeLimit', 3600)
start_time = time.time()
model.optimize()

if model.status == GRB.OPTIMAL:
    end_time = time.time()
    runtime = round(end_time - start_time, 2)
    objective = round(model.objVal)
    results['relaxations']['FYL'] = {'time': runtime, 'objective': objective}
elif model.status == GRB.TIME_LIMIT:
    end_time = time.time()
    runtime = round(end_time - start_time, 2)
    results['relaxations']['FYL'] = {'time': runtime, 'status': 'time_limit'}
elif model.status == GRB.MEM_LIMIT:
    end_time = time.time()
    runtime = round(end_time - start_time, 2)
    results['relaxations']['FYL'] = {'time': runtime, 'status': 'soft_memory_limit'}
else:
    results['relaxations']['FYL'] = {'status': 'not_solved'}




#AJL
model = Model('AJL_LP_Relaxation')
model.params.method = 2
model.params.crossover = 0
model.params.SoftMemLimit = 16
x = model.addVars(n, n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='x')
y = model.addVars(n, n, n, n, vtype=GRB.CONTINUOUS, lb=0, name='y')


model.update()

model.setObjective(
    sum(c[i,j,k,l] * y[i,j,k,l] for i in range(n) for j in range(n) for k in range(n) for l in range(n)),
    GRB.MINIMIZE)

for i in range(n):
    model.addConstr(sum(x[i,j] for j in range(n)) == 1)
for j in range(n):
    model.addConstr(sum(x[i,j] for i in range(n)) == 1)

for j in range(n):
    for k in range(n):
        for l in range(n):
            model.addConstr(sum(y[i,j,k,l] for i in range(n)) == x[k,l])

for i in range(n):
    for k in range(n):
        for l in range(n):
            model.addConstr(sum(y[i,j,k,l] for j in range(n)) == x[k,l])

for i in range(n):
    for j in range(n):
        for k in range(n):
            for l in range(n):
                model.addConstr(y[i,j,k,l] == y[k,l,i,j])

model.update()
model.setParam('TimeLimit', 3600)
start_time = time.time()
model.optimize()

if model.status == GRB.OPTIMAL:
    end_time = time.time()
    runtime = round(end_time - start_time, 2)
    objective = round(model.objVal)
    results['relaxations']['AJL'] = {'time': runtime, 'objective': objective}
elif model.status == GRB.TIME_LIMIT:
    end_time = time.time()
    runtime = round(end_time - start_time, 2)
    results['relaxations']['AJL'] = {'time': runtime, 'status': 'time_limit'}
elif model.status == GRB.MEM_LIMIT:
    end_time = time.time()
    runtime = round(end_time - start_time, 2)
    results['relaxations']['AJL'] = {'time': runtime, 'status': 'soft_memory_limit'}
else:
    results['relaxations']['AJL'] = {'status': 'not_solved'}





#KBL
model = Model('KBL_LP_Relaxation')
model.params.method = 2
model.params.crossover = 0
model.params.SoftMemLimit = 16
x = model.addVars(n, n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='x')
w = model.addVars(n, n, vtype=GRB.CONTINUOUS, lb=0, name='w')
model.update()

model.setObjective(sum(w[i,j] for i in range(n) for j in range(n)), GRB.MINIMIZE)

for i in range(n):
    model.addConstr(sum(x[i,j] for j in range(n)) == 1)
for j in range(n):
    model.addConstr(sum(x[i,j] for i in range(n)) == 1)

for i in range(n):
    for j in range(n):
        model.addConstr(w[i,j] >= sum(c[i,j,k,l] * x[k,l] for k in range(n) for l in range(n)) - A[i,j] * (1 - x[i,j]))

model.update()
model.setParam('TimeLimit', 3600)
start_time = time.time()
model.optimize()

if model.status == GRB.OPTIMAL:
    end_time = time.time()
    runtime = round(end_time - start_time, 2)
    objective = round(model.objVal)
    results['relaxations']['KBL'] = {'time': runtime, 'objective': objective}
elif model.status == GRB.TIME_LIMIT:
    end_time = time.time()
    runtime = round(end_time - start_time, 2)
    results['relaxations']['KBL'] = {'time': runtime, 'status': 'time_limit'}
elif model.status == GRB.MEM_LIMIT:
    end_time = time.time()
    runtime = round(end_time - start_time, 2)
    results['relaxations']['KBL'] = {'time': runtime, 'status': 'soft_memory_limit'}
else:
    results['relaxations']['KBL'] = {'status': 'not_solved'}




#XYL
model = Model('XYL_LP_Relaxation')
model.params.method = 2
model.params.crossover = 0
model.params.SoftMemLimit = 16
x = model.addVars(n, n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='x')
z = model.addVars(n, n, vtype=GRB.CONTINUOUS, name='z')
model.update()

model.setObjective(
    sum(z[i,j] + c[i,j,i,j] * x[i,j] for i in range(n) for j in range(n)),
    GRB.MINIMIZE)

for i in range(n):
    model.addConstr(sum(x[i,j] for j in range(n)) == 1)
for j in range(n):
    model.addConstr(sum(x[i,j] for i in range(n)) == 1)

for i in range(n):
    for j in range(n):
        double_sum = sum(c[i,j,k,l] * x[k,l] for k in range(n) if k != i for l in range(n) if l != j)
        model.addConstr(z[i,j] >= double_sum - U[i,j] * (1 - x[i,j]))
        model.addConstr(z[i,j] >= L[i,j] * x[i,j])

model.update()
model.setParam('TimeLimit', 3600)
start_time = time.time()
model.optimize()

if model.status == GRB.OPTIMAL:
    end_time = time.time()
    runtime = round(end_time - start_time, 2)
    objective = round(model.objVal)
    results['relaxations']['XYL'] = {'time': runtime, 'objective': objective}
elif model.status == GRB.TIME_LIMIT:
    end_time = time.time()
    runtime = round(end_time - start_time, 2)
    results['relaxations']['XYL'] = {'time': runtime, 'status': 'time_limit'}
elif model.status == GRB.MEM_LIMIT:
    end_time = time.time()
    runtime = round(end_time - start_time, 2)
    results['relaxations']['XYL'] = {'time': runtime, 'status': 'soft_memory_limit'}
else:
    results['relaxations']['XYL'] = {'status': 'not_solved'}





#GLL
model = Model('GLL_LP_Relaxation')
model.params.method = 2
model.params.crossover = 0
model.params.SoftMemLimit = 16
x = model.addVars(n, n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='x')
g = model.addVars(n, n, vtype=GRB.CONTINUOUS, lb=0, name='g')
model.update()

model.setObjective(
    sum(g[i,j] + (c[i,j,i,j] + L[i,j]) * x[i,j] for i in range(n) for j in range(n)),
    GRB.MINIMIZE)

for i in range(n):
    model.addConstr(sum(x[i,j] for j in range(n)) == 1)
for j in range(n):
    model.addConstr(sum(x[i,j] for i in range(n)) == 1)

for i in range(n):
    for j in range(n):
        double_sum = sum(c[i,j,k,l] * x[k,l] for k in range(n) for l in range(n))
        model.addConstr(g[i,j] >= double_sum - A[i,j] * (1 - x[i,j]) - (c[i,j,i,j] + L[i,j]) * x[i,j])

model.update()
model.setParam('TimeLimit', 3600)
start_time = time.time()
model.optimize()

if model.status == GRB.OPTIMAL:
    end_time = time.time()
    runtime = round(end_time - start_time, 2)
    objective = round(model.objVal)
    results['relaxations']['GLL'] = {'time': runtime, 'objective': objective}
elif model.status == GRB.TIME_LIMIT:
    end_time = time.time()
    runtime = round(end_time - start_time, 2)
    results['relaxations']['GLL'] = {'time': runtime, 'status': 'time_limit'}
elif model.status == GRB.MEM_LIMIT:
    end_time = time.time()
    runtime = round(end_time - start_time, 2)
    results['relaxations']['GLL'] = {'time': runtime, 'status': 'soft_memory_limit'}
else:
    results['relaxations']['GLL'] = {'status': 'not_solved'}



output_filename = filename.replace('.dat', '_results.json')
with open(output_filename, 'w') as f:
    json.dump(results, f, indent=1)
