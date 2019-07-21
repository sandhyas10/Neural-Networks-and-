import time
import numpy as np

def pca_naive(X, K):
    """
    PCA -- naive version

    Inputs:
    - X: (float) A numpy array of shape (N, D) where N is the number of samples,
         D is the number of features
    - K: (int) indicates the number of features you are going to keep after
         dimensionality reduction

    Returns a tuple of:
    - P: (float) A numpy array of shape (K, D), representing the top K
         principal components
    - T: (float) A numpy vector of length K, showing the score of each
         component vector
    """

    ###############################################
    #TODO: Implement PCA by extracting eigenvector#
    ###############################################
    N=X.shape[0]
    corr=np.corrcoef(X.transpose())
    P=[]
    T=[]
    pcas=[]

    eigenval,eigenvecs=np.linalg.eig(corr)
    for i in range(len(eigenval)):
        pcas.append([np.abs(eigenval[i]),eigenvecs[:,i]])
    
    pcas=sorted(pcas,reverse=True)
    for i in range(K-1):
        P.append(pcas[i][1])
        T.append(pcas[i][0])
    ###############################################
    #              End of your code               #
    ###############################################
    return (P, T)
