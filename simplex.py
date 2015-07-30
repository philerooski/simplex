import numpy as np
np.set_printoptions(precision=3)

EPSILON = 1e-3

A = np.array([[1, 2, 3, 0], 
                [-1, 2, 6, 0], 
                [0, 4, 9, 0],
                [0, 0, 3, 1]])
c = np.array([1, 2, 3, 0])
b = np.array([3, 2, 5, 1]).T

def get_initial_bfs(A, b, c):
    '''Find an initial basic feasible solution to the LP problem.
        This requires applying the simplex method to an auxilary LP problem.'''
    aux_A = np.concatenate((A, np.identity(len(A))), axis=1)
    aux_c = np.concatenate((np.zeros(len(A)), np.ones(len(A))), axis=1)
    aux_basic_indices = np.array(range(len(A.T), len(aux_A.T)))
    basic_indices, basic_cost_coeff, neg_cost, reduced_costs, tableau \
            = simplex(aux_A, b, aux_c, aux_basic_indices)
    if neg_cost > 0: # unable to set all artificial variables to 0
        raise Exception("No feasible solutions exist")
    if any(basic_indices >= len(A.T)): # remove any leftover artificial variables
        artificial_vars = filter(lambda i: basic_indices[i] >= len(A.T), 
                range(len(basic_indices)))
        for var in artificial_vars:
            row = A.T[1:][var]
            for i in row:
                if i not in basic_indices and i != 0 and i < len(A.T):
                    basic_indices[var] = i
            if basic_indices[var] >= len(A.T):
                # if no replacement found, there is a 
                # linear dependence in the row space of A 
                basic_indices = np.delete(basic_indices, var)
                b = np.delete(b, var)
                A = np.delete(A, var, axis=0)
    tableau = np.delete(tableau.T[1:], range(len(A.T), len(aux_A.T)), 0).T
    #TODO: return all relevent information, optimize methods
    return basic_indices, A, b, tableau

def simplex(A, b, c, basic_indices):
    basis = get_cols(A, basic_indices)
    basis_inverse = np.linalg.inv(basis)
    tableau = np.concatenate((np.atleast_2d(np.dot(basis_inverse, b)).T, 
        np.dot(basis_inverse, A)), axis=1) 
    basic_cost_coeff = get_cols(c, basic_indices)
    get_reduced_costs = lambda b, t: c.T - np.dot(b, t.T[1:].T)[0]
    get_neg_cost = lambda b, t: -np.dot(b, t.T[0])
    neg_cost = get_neg_cost(basic_cost_coeff, tableau)
    reduced_costs = get_reduced_costs(basic_cost_coeff, tableau)
    iterations = 0
    while not all(reduced_costs > np.zeros(len(reduced_costs)) - EPSILON):
        entering_index = 0
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
        exiting_index = ratios.index(min(ratios))
        for row in range(len(tableau)):
            if row != exiting_index:
                tableau[row] -= (tableau[exiting_index] * 
                entering_col[row] / entering_col[exiting_index])
            else:
                tableau[row] /= entering_col[exiting_index] 
        basic_indices[exiting_index] = entering_index
        basic_cost_coeff = get_cols(c, basic_indices)
        neg_cost -= (reduced_costs[entering_index] 
            * tableau.T[0].T[exiting_index])
        reduced_costs -= (reduced_costs[entering_index] 
            * tableau.T[1:].T[exiting_index])
        iterations += 1
        if iterations > 100:
            break
    return basic_indices, basic_cost_coeff, neg_cost, reduced_costs, tableau
            

def get_cols(M, indices): 
    return np.column_stack([M.T[i] for i in indices])

def optimize(A, b, c):
    basic_indices, A, b, tableau = get_initial_bfs(A, b, c)
    return simplex(A, b, c, basic_indices)

print optimize(A, b, c)
