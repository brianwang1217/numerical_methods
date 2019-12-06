import numpy as np
import matplotlib.pyplot as plt

''' Page 4 '''
def MyLU(A):

    # The upper triangular matrix U is saved in the upper part of the matrix M (including the diagonal)
    # The lower triangular matrix L is saved in the lower part of the matrix M (not including the diagonal)
    # Do NOT use `scipy.linalg.lu`!
    # You should not use pivoting

    M = A.copy()
    for i in range(len(A)):
        M[i+1:,i] = M[i+1:,i]/M[i,i]
        M[i+1:,i+1:] -= np.outer(M[i+1:,i],M[i,i+1:])
    return M

def MyTriangularSolve(M,b):

    # A = LU (L and U are stored in M)
    # A x = b (given A and b, find x)
    # M is a 2D numpy array
    # The upper triangular matrix U is stored in the upper part of the matrix M (including the diagonal)
    # The lower triangular matrix L is stored in the lower part of the matrix M (not including the diagonal)
    # b is a 1D numpy array
    # x is a 1D numpy array
    # Do not use `scipy.linalg.solve_triangular`
    n = len(M)
    x = np.zeros(n)
    y = np.zeros(n)

    for i in range(0, n):
        tmp = b[i]
        for j in range(0, i):
            tmp -= y[j]*M[i,j]
        y[i]=tmp

    for i in range(n-1, -1, -1):
        tmp = y[i]
        for j in range(i+1, n):
            tmp -= x[j]*M[i,j]
        x[i] = tmp/M[i,i]

    return x

def SolveLinsys(kff,kpf,kfp,kpp,up,pf):

    # Use MyLU and MyTriangularSolve
    kff = MyLU(kff)
    uf = MyTriangularSolve(kff, pf - kfp @ up)
    pp = kpp @ up + kpf @ uf
    return uf, pp


# Compute uf and pp for this particular spring system by calling
# your functions with variables listed under 'INPUT'
uf, pp = SolveLinsys(kff, kpf, kfp, kpp, up, pf)

''' Page 5 '''
def MyLU(A):

    # The upper triangular matrix U is saved in the upper part of the matrix M (including the diagonal)
    # The lower triangular matrix L is saved in the lower part of the matrix M (not including the diagonal)
    # Do NOT use `scipy.linalg.lu`!
    # You should not use pivoting

    M = A.copy()
    for i in range(len(A)):
        M[i+1:,i] = M[i+1:,i]/M[i,i]
        M[i+1:,i+1:] -= np.outer(M[i+1:,i],M[i,i+1:])
    return M

def MyTriangularSolve(M,b):

    # A = LU (L and U are stored in M)
    # A x = b (given A and b, find x)
    # M is a 2D numpy array
    # The upper triangular matrix U is stored in the upper part of the matrix M (including the diagonal)
    # The lower triangular matrix L is stored in the lower part of the matrix M (not including the diagonal)
    # b is a 1D numpy array
    # x is a 1D numpy array
    # Do not use `scipy.linalg.solve_triangular`
    n = len(M)
    x = np.zeros(n)
    y = np.zeros(n)

    for i in range(0, n):
        tmp = b[i]
        for j in range(0, i):
            tmp -= y[j]*M[i,j]
        y[i]=tmp

    for i in range(n-1, -1, -1):
        tmp = y[i]
        for j in range(i+1, n):
            tmp -= x[j]*M[i,j]
        x[i] = tmp/M[i,i]

    return x

def SolveLinsys(kff,kpf,kfp,kpp,up,pf):

    # Use MyLU and MyTriangularSolve
    kff = MyLU(kff)
    uf = MyTriangularSolve(kff, pf - kfp @ up)
    pp = kpp @ up + kpf @ uf
    return uf, pp


# Compute uf and pp for this particular spring system by calling
# your functions with variables listed under 'INPUT'
uf, pp = SolveLinsys(kff, kpf, kfp, kpp, up, pf)

''' Page 6 '''
kff = MyLU(kff)
uf1 = MyTriangularSolve(kff, pf[:,0] - kfp @ up)
plotTruss(uf1)

uf2 = MyTriangularSolve(kff, pf[:,1] - kfp @ up)
plotTruss(uf2)
