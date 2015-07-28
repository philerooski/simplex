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
    aux_A = np.concatenate((A, np.identity(len(A))), axis=1)
    aux_c = np.concatenate((np.zeros(len(A)), np.ones(len(A))), axis=1)
    aux_basic_indices = np.array(range(len(A.T), len(aux_A.T)))
    cost, reduced_costs = simplex(aux_A, b, aux_c, aux_basic_indices)

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
        entering_col = tableau.T[entering_index]
        ratios = []
        for i in range(len(basic_indices)):
            if entering_col[i] > 0:
                ratios.append(tableau.T[0][i] / entering_col[i])
            else: 
                ratios.append(float('inf'))
        exiting_basic_index = ratios.index(min(ratios))
        for row in range(len(tableau)):
            if row != exiting_basic_index:
                tableau[row] -= (tableau[exiting_basic_index] * 
                entering_col[row] / entering_col[exiting_basic_index])
            else:
                tableau[row] /= entering_col[exiting_basic_index] 
        basic_indices[exiting_basic_index] = entering_index
        basic_cost_coeff = get_cols(c, basic_indices)
        neg_cost = get_neg_cost(basic_cost_coeff, tableau)
        reduced_costs -= (tableau.T[1:].T[exiting_basic_index] * 
                (reduced_costs[entering_index] 
                / entering_col[exiting_basic_index]))
        iterations += 1
        break
        #TODO: convergence is too slow
    return 1, 2
            

def get_cols(M, indices): 
    return np.column_stack([M.T[i] for i in indices])

get_initial_bfs(A, b, c)
