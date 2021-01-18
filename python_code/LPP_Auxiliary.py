"""
Created on Mon Jan  18 2021

%%%%%%%%%%%%%%%%%%%% Auxiliary functions to perform LPP %%%%%%%%%%%%%%%%%%%%

@author: Wenqing Hu (Missouri S&T)
"""

# k-nearest neighbor classfication
# given test data x and label y, find in a training set (X, Y) the k-nearest points x1,...,xk to x, and classify x as majority vote on y1,...,yk
# if the classification is correct, return 1, otherwise return 0
def knn(x_test, y_test, X_train, Y_train, k):
    m = len(Y_train)
    if k>m:
        k=m
    # find the first k-nearest neighbor
    dist = [np.linalg.norm(np.array(x_test)-np.array(X_train[i])) for i in range(m)]
    indexes, dist_sort = zip(*sorted(enumerate(dist), key=itemgetter(1))) 
    # do a majority vote on the first k-nearest neighbor
    label = [Y_train[indexes[_]] for _ in range(k)]
    vote = pd.value_counts(label)
    # class_predict is the predicted label based on majority vote
    class_predict = vote.index[0]
    if class_predict == y_test:
        isclassified = 1
    else:
        isclassified = 0
    return isclassified, class_predict


# solve the laplacian embedding, given data set X={x1,...,xm}, the graph laplacian L and degree matrix D    
def LPP(X, L, D):
    # turn X, L, D into arrays
    X = np.array(X)
    L = np.array(L)
    D = np.array(D)
    # calculate mtx_L = X' * L * X
    mtx_L = np.matmul(np.matmul(X.T, L), X)
    # calculate mtx_D = X' * D * X
    mtx_D = np.matmul(np.matmul(X.T, D), X)
    # solve the generalized eigenvalue problem mtx_L W = LAMBDA mtx_D W
    LAMBDA, W = eigh(mtx_L, mtx_D, eigvals_only=False)
    # sort the eigenvalues in a descending order
    SORT_ORDER, LAMBDA = zip(*sorted(enumerate(LAMBDA), key=itemgetter(1), reverse=False)) 
    # reorder the generalized eigenvector matrix W according to SORT_ORDER
    W = [W[SORT_ORDER[_]] for _ in range(len(SORT_ORDER))]
    return W, LAMBDA 
    
 
# construct the graph laplacian L and the degress matrix D from the given affinity matrix S 
def graph_laplacian(S):
    # first turn S into an array
    S = np.array(S)
    # compute the D matrix
    D = np.diag(sum(S, 0))
    L = D - S
    return L, D


# given a set of data points X={x1,...,xm} with label Y={y1,...,ym}, construct their supervised affinity matrix S for LPP
def affinity_supervised(X, Y, between_class_affinity):
    # original distances squares between xi and xj
    f_dist1 = cdist(X, X, 'euclidean')
    # heat kernel size
    mdist = np.mean(f_dist1) 
    h = -np.log(0.15)/mdist
    S1 = np.exp(-h*f_dist1)
    # utilize supervised info
    # first turn Y into a 2-d array
    Y = [[Y[_]] for _ in range(len(Y))]
    id_dist = cdist(Y, Y, 'euclidean')
    S2 = S1 
    for i in range(len(X)):
        for j in range(len(X)):
            if id_dist[i][j] != 0:
                S2[i][j] = between_class_affinity
    # obtain the supervised affinity S
    S = S2
    return S
