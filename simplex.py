import numpy as np

EPSILON = 1e-6

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
    neg_cost = -np.dot(basic_cost_coeff, tableau.T[0])
    reduced_costs = c.T - np.dot(basic_cost_coeff, tableau.T[1:].T)[0]
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
        #TODO: apply pivot transformations to simplex tableau
        break
    return 1, 2
            

def get_cols(M, indices): 
    return np.column_stack([M.T[i] for i in indices])

get_initial_bfs(A, b, c)
