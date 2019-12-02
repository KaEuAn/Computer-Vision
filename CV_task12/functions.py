import numpy as np


def triangulate_point(x1,x2,P1,P2):
    """ Point pair triangulation from 
        least squares solution. """
        
    M = np.zeros((6,6))
    M[:3,:4] = P1
    M[3:,:4] = P2
    M[:3,4] = -x1
    M[3:,5] = -x2

    U,S,V = np.linalg.svd(M)
    X = V[-1,:4]

    return X / X[3]


def triangulate(x1,x2,P1,P2):
    """    Two-view triangulation of points in 
        x1,x2 (3*n homog. coordinates). """
        
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    X = [ triangulate_point(x1[:,i],x2[:,i],P1,P2) for i in range(n)]
    return np.array(X).T


def skew(a):
    """ Skew matrix A such that a x v = Av for any v. """

    return np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])

def compute_P_from_essential(E):
    """    Computes the second camera matrix (assuming P1 = [I 0]) 
        from an essential matrix. Output is a list of four 
        possible camera matrices. """
    
    # make sure E is rank 2
    U,S,V = np.linalg.svd(E)
    if np.linalg.det(np.dot(U,V))<0:
        V = -V
    E = np.dot(U,np.dot(np.diag([1,1,0]),V))    
    
    # create matrices (Hartley p 258)
    Z = skew([0,0,-1])
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    
    # return all four solutions
    P2 = [np.vstack((np.dot(U, np.dot(W,   V)).T,  U[:,2])).T,
          np.vstack((np.dot(U, np.dot(W,   V)).T, -U[:,2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T,  U[:,2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:,2])).T]

    return P2


def F_from_ransac(x1,x2,model,maxiter=5000,match_theshold=1e-6):
    """ Robust estimation of a fundamental matrix F from point 
        correspondences using RANSAC (ransac.py from
        http://www.scipy.org/Cookbook/RANSAC).

        input: x1,x2 (3*n arrays) points in hom. coordinates. """

    from ransac import ransac

    data = np.vstack((x1,x2))
    
    # compute F and return with inlier index
    F,ransac_data = ransac(data.T,model,8,maxiter,match_theshold,20,return_all=True)
    return F, ransac_data['inliers']
