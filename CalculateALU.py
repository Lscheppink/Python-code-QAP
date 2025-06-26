from gurobipy import *
import numpy as np

def calculate_A(c):
    n=len(c)
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            model = Model('A constants')
            model.Params.LogToConsole=0

            x = model.addVars(n, n, name='x', vtype=GRB.CONTINUOUS, lb=0, ub=1)
            model.update()

            model.addConstrs(sum(x[i,j] for j in range(n)) == 1 for i in range(n))
            model.addConstrs(sum(x[i,j] for i in range(n)) == 1 for j in range(n))
            model.update()

            obj_fn = sum(c[i,j,k,l] * x[k,l] for k in range(n) for l in range(n))
            model.setObjective(obj_fn, GRB.MAXIMIZE)
            model.optimize()
            if model.status == GRB.OPTIMAL:
                A[i,j] = model.objVal
                # X_sol = np.array([[x[i, j].x for j in range(n)] for i in range(n)])
                # print('Assignment matrix')
                # print(X_sol)
            else:
                print('Could not solve to optimality.')
    return A

def calculate_L(c):
    n=len(c)
    L = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            model = Model('L constants')
            model.Params.LogToConsole=0

            # x = model.addVars(n, n, name='x', vtype=GRB.BINARY)
            x = model.addVars(n, n, name='x', vtype=GRB.CONTINUOUS, lb=0, ub=1)
            model.update()

            model.addConstrs(sum(x[i,j] for j in range(n)) == 1 for i in range(n))
            model.addConstrs(sum(x[i,j] for i in range(n)) == 1 for j in range(n))

            model.addConstr(x[i,j] == 1)
            model.update()

            obj_fn = sum(c[i,j,k,l] * x[k,l]
                         for k in range(n) if k != i
                         for l in range(n) if l != j)
            model.setObjective(obj_fn, GRB.MINIMIZE)
            model.optimize()
            if model.status == GRB.OPTIMAL:
                L[i,j] = model.objVal
            else:
                print('Could not solve to optimality.')
    return L

def calculate_U(c):
    n=len(c)
    U = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            model = Model('U constants')
            model.Params.LogToConsole=0

            # x = model.addVars(n, n, name='x', vtype=GRB.BINARY)
            x = model.addVars(n, n, name='x', vtype=GRB.CONTINUOUS, lb=0, ub=1)
            model.update()

            model.addConstrs(sum(x[i,j] for j in range(n)) == 1 for i in range(n))
            model.addConstrs(sum(x[i,j] for i in range(n)) == 1 for j in range(n))
            model.addConstr(x[i,j] == 0)
            model.update()

            obj_fn = sum(c[i,j,k,l] * x[k,l]
                         for k in range(n) if k != i
                         for l in range(n) if l != j)
            model.setObjective(obj_fn, GRB.MAXIMIZE)
            model.optimize()
            if model.status == GRB.OPTIMAL:
                U[i,j] = model.objVal
                # X_sol = np.array([[x[i, j].x for j in range(n)] for i in range(n)])
                # print(f'Assignment matrix {i,j}')
                # print(X_sol)
            else:
                print('Could not solve to optimality.')
    return U




