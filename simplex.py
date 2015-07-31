''' simplex.py
An implementation of the full tableau simplex algorithm.
Author: Phil Snyder
'''

import numpy as np
EPSILON = 1e-6

# a somewhat trivial example, since variables A1, A2, A3 are linearly
# independent (and thus the solution to Ax=b is uniquely determined).
A = np.array([[1, 2, 2, 1, 0, 0],
                [2, 1, 2, 0, 1, 0],
                [2, 2, 1, 0, 0, 1]])
b = np.array([20, 20, 20])
c = np.array([-10, -12, -12, 0, 0, 0])

def get_initial_bfs(A, b, c):
    '''Find an initial basic feasible solution to the LP problem.
    This requires applying the simplex method to an auxilary LP problem.'''
    aux_A = np.concatenate((A, np.identity(len(A))), axis=1)
    aux_c = np.concatenate((np.zeros(len(A.T)), np.ones(len(A))), axis=1)
    aux_basic_indices = np.array(range(len(A.T), len(aux_A.T)))
    basic_indices, neg_cost, reduced_costs, tableau \
            = simplex(aux_A, b, aux_c, aux_basic_indices)
    if neg_cost != 0: # unable to set all artificial variables to 0
        raise Exception("No feasible solutions exist")
    if any(basic_indices >= len(A.T)): # remove any leftover artificial variables
        artificial_vars = filter(lambda i: basic_indices[i] >= len(A.T), 
                range(len(basic_indices)))
        to_delete = []
        for var in artificial_vars:
            row = A.T[1:][var]
            for i in row:
                if i not in basic_indices and i != 0 and i < len(A.T):
                    basic_indices[var] = i
            if basic_indices[var] >= len(A.T):
                # if no replacement found, there is a 
                # linear dependence in the row space of A 
                to_delete.append(basic_indices[var])
        for i in to_delete:
            var = basic_indices.tolist().index(i)
            basic_indices = np.delete(basic_indices, var)
            b = np.delete(b, var)
            A = np.delete(A, var, axis=0)
    tableau = np.delete(tableau.T[1:], range(len(A.T), len(aux_A.T)), 0).T
    return basic_indices, A, b, tableau

def simplex(A, b, c, basic_indices):
    '''Run (up to 1000) iterations of the simplex method until 
    the reduced costs of all variables are less than EPSILON'''
    basis = get_cols(A, basic_indices)
    basis_inverse = np.linalg.inv(basis)
    tableau = np.concatenate((np.atleast_2d(np.dot(basis_inverse, b)).T, 
        np.dot(basis_inverse, A)), axis=1) 
    basic_cost_coeff = get_cols(c, basic_indices)[0]
    neg_cost = -np.dot(basic_cost_coeff, tableau.T[0])
    reduced_costs = c - np.dot(basic_cost_coeff, tableau.T[1:].T)
    iterations = 0
    while not all(reduced_costs > np.zeros(len(reduced_costs)) - EPSILON):
        # there exists a basic feasible direction we can head in that will 
        # reduce our cost function c'x
        entering_index = 0
        # bland's rule (smallest subscript) is used for pivoting
        for index in range(len(reduced_costs)):
            if reduced_costs[index] < -EPSILON:
                entering_index = index
                break
        entering_col = tableau.T[1:][entering_index]
        ratios = []
        for i in range(len(basic_indices)):
            if entering_col[i] > 0:
                ratios.append(tableau.T[0][i] / entering_col[i])
            else: 
                ratios.append(float('inf'))
        if not any(np.array(ratios) < float('inf')):
            raise Exception("Optimal cost is -infinity")
        exiting_index = ratios.index(min(ratios))
        for row in range(len(tableau)):
            if row != exiting_index:
                tableau[row] -= (tableau[exiting_index] * 
                entering_col[row] / entering_col[exiting_index])
            else:
                tableau[row] /= entering_col[exiting_index] 
        basic_indices[exiting_index] = entering_index
        neg_cost -= (reduced_costs[entering_index] 
            * tableau.T[0].T[exiting_index])
        reduced_costs -= (reduced_costs[entering_index] 
            * tableau.T[1:].T[exiting_index])
        iterations += 1
        if iterations > 1000:
            print "Maximum iterations exceeded"
            break 
    return basic_indices, neg_cost, reduced_costs, tableau
            
def get_cols(M, indices): 
    '''Returns the columns of M at the specified indices'''
    return np.column_stack([M.T[i] for i in indices])

def optimize(A, b, c):
    '''The primary function of simplex.py. Accepts a matrix A in R^m*n of 
    constraint coefficients, a vector b in R^m of equalities, and a vector 
    c in R^n of cost coefficients. The simplex method returns a solution 
    to Ax=b that minimizes the objective function c'x. All parameters are
    np.array. returns cost, basic indices and their values'''
    basic_indices, A, b, tableau = get_initial_bfs(A, b, c)
    basic_indices, neg_cost, reduced_costs, tableau = simplex(A, b, c, basic_indices)
    return -neg_cost, basic_indices, tableau.T[0]
