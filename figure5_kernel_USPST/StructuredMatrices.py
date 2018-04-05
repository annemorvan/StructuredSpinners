# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:33:56 2016

@author: amorvan

This file contains all functions related to structured matrices.
* How to construct them
* How to perform efficiently matrix-vector product
* How to test them with A = B3B2B1 combinations

Algorithms described in:

"FAST ALGORITHMS TO COMPUTE MATRIX-VECTOR
PRODUCTS FOR PASCAL MATRICES" from Tang and al.

and "Structured Transforms for Small-Foorprint Deep Learning"
from Sindhwani and al. for Kronecker matrices

"""


from __future__ import division

import numpy as np
# Trying out FFTW would be even better. Unfortunately Python bindings at
# https://hgomersall.github.io/pyFFTW/ do not support DCT methods, perhaps because numpy.fft
# does not implement it either.
from scipy.fftpack import dct
from scipy.linalg import circulant
from scipy.linalg import hadamard
from scipy.linalg import toeplitz
from scipy.linalg import hankel
import time
# Currently not used, not installed on Tamas' Mac.
# from scipy.misc import toimage 
import cmath
import math
import matplotlib.pyplot as plt
import sys

# Various Fast Hadamard Transform implementations, see README how to build the
# C code they call.
#import fht
import ffht
#from hadamard_transform import fwht_copy, fwht_inplace
#from hadamard_transform_spiral import fwht_spiral_copy, fwht_spiral_inplace

#==============================================================================
# Structured matrices computation
#==============================================================================


# RANDOM GAUSSIAN MATRIX
def random(d): 
    """
    Returns a d x d random matrix which every entries are draw from a Gaussian N(0,1)
    distribution
    """       
    A = np.random.normal(0.0, 1.0, d*d).reshape(d,d)
    return A  
 


# SKEW-CIRCULANT MATRIX
def skew_circulant(c):
    """
    Returns a Skew-circulant matrix from the first column c in parameter.
    It is like a circulant matrix but the upper diagonal entries are flipped.
    
    REMARK: JUST FOR TESTING, USELESS WHEN PERFORMING MATRIX-VECTOR PRODUCT
    BECAUSE THE SKEW-CIRCULANT MATRIX IS NOT BUILD EXPLICITLY.
    """
    SkewC = circulant(c)
    #print('C')
    #print(SkewC)
    #TODO: to be optimized
    for i in xrange(n):
        for j in xrange(n):
            if i < j :
                SkewC[i,j] *= -1
    return SkewC
    


# # HD3 HD2 HD1      
#def hadamardCrossPolytope(d, nblocks = 3):
#    """
#    Returns a structured matrix from 
#    "Practical and Optimal LSH for Angular Distance ", Alexandr Andoni and al.
#    A = H D_3 H D_2 H D_1
#   
#    H is the Hadamard matrix.
#    
#    Di are diagonal random matrices which entries are drawn independently 
#    randomly from {-1, 1}.
#    
#    One diagonal matrix is represented by its diagonal coefficients which is
#    one of rows of D with dimension nblocks x d
#    
#    Parameters: 
#    * d is data dimensionality
#    * nblocks is the number of blocks HDi, by default, nblocks = 3
#    
#    REMARK: JUST FOR TESTING, USELESS WHEN PERFORMING MATRIX-VECTOR PRODUCT
#    BECAUSE H IS NOT BUILD EXPLICITLY.
#    """
#    H = hadamard(d) # H is not normalized
#
#    # D is a matrix containing diag coefficients, nblocks x d
#    D = np.ones((nblocks, d), dtype = np.int) 
#    # D[i,j] is the j-th coefficient for D_i matrix
#    #TODO: possible to be faster ?
#    for i in xrange(nblocks):
#        for j in xrange(d):
#            coef = np.random.rand()
#            if coef <= 0.5:
#                D[i,j] *= -1    
#
#    return H, D



# Di
#def compute_diagonal_matrices(dim, nblocks):
#    """
#    Returns random discrete diagonal matrices used in HD1HD2H3 by computing only each diagonal.
#    One diagonal matrix is represented by its diagonal coefficients which is
#    one of rows of D with dimension nblocks x d
#    
#    Di are diagonal random matrices which entries are drawn independently 
#    randomly from {-1, 1}.
#    
#    REMARK: FUNCTION TO USE IN PRACTICE0
#    """
#    # D is a matrix containing diag coefficients, nblocks x dim
#    D = np.ones((nblocks, dim), dtype = np.int) 
#    # D[i,j] is the j-th coefficient for D_i matrix
#    #TODO: possible to be faster ?
#    for i in xrange(nblocks):
#        for j in xrange(dim):
#            coef = np.random.rand()
#            if coef <= 0.5:
#                D[i,j] *= -1    
#    return D

def sign11(x):
    x_bin = np.ones(x.shape, dtype = 'int')
    x_bin[x < 0] = -1
    return x_bin
    
def compute_diagonal_matrices(dim, nblocks):
    """
    Returns random discrete diagonal matrices used in HD1HD2H3 by computing only each diagonal.
    One diagonal matrix is represented by its diagonal coefficients which is
    one of rows of D with dimension nblocks x d
    
    Di are diagonal random matrices which entries are drawn independently 
    randomly from {-1, 1}.
    
    REMARK: FUNCTION TO USE IN PRACTICE0
    """
    # D is a matrix containing diag coefficients, nblocks x dim
    D = np.random.normal(0.0, 1.0, dim * nblocks)
    D = sign11(D)
    return D.reshape((nblocks, dim))

## D
#def compute_one_diagonal(dim):
#    """
#    Returns diagonal entries from a discrete random diagonal matrix.
#    Entries drawn independently randomly from {-1, 1}.
#    """
#    D = np.ones(dim) 
#    #TODO: possible to be faster ?
#    for j in xrange(dim):
#        coef = np.random.rand()
#        if coef <= 0.5:
#            D[j] *= -1    
#    return D

def compute_one_diagonal(dim):
    """
    Returns diagonal entries from a discrete random diagonal matrix.
    Entries drawn independently randomly from {-1, 1}.
    """
    D = np.random.normal(0.0, 1.0, dim)
    #TODO: possible to be faster ?  
    return sign11(D)

# S
def compute_feature_hashing(dim1, dim2):
    """
    Computes feature hashing + dimensionality reduction with dim1 << dim2
    by returning S.
    S is a random sparse dim1 x dim2 matrix whose columns
    have ONE non-zero +-1 entry sampled  uniformly
    """
    S = np.zeros((dim1, dim2), dtype = np.int) 
    
    # for each column
    for i in xrange(dim2):
        # choose one non-zero entry
        non_zero_entry = np.random.choice(xrange(dim1))
        # +1 or -1?
        coef = np.random.rand()
        if coef <= 0.5:
            S[non_zero_entry, i] = -1  
        else:
            S[non_zero_entry, i] = 1  
    return S



#def kronecker(m, de, discrete = False):
#    """
#    compute a discrete (+1/-1) or a Gaussian Kronecker matrix
#    """
#    
#    t2 = time.time()
#    A_list = []
#    
#    if discrete:
#        H = 1.0 / math.sqrt(2) * hadamard(de)
#        
#    for i in xrange(0, m):
#        
#        if discrete:
#            A = np.copy(H)
#            # random multiplication *-1 on rows or columns
#            # multiplication *-1 ?
#            r = (np.random.rand(), np.random.randint(de), np.random.rand(), np.random.randint(de))
#            if r[0] > 0.5:
#                A[r[1], :] *= -1
#            if r[2] > 0.5:
#                A[:, r[3]] *= -1
#        else:
#            A = np.random.uniform(0.0, 1.0, 2*2).reshape(2,2)
#            A, _ = np.linalg.qr(A)
#            
#        #print('Ai', A)
#        A_list.append(A)  
#    t2 = time.time() - t2
#    print('time to compute A_list', t2)
#    K = A_list[0]    
#    for i in xrange(1, m):
#        K = np.kron(K, A_list[i])
#    return K, A_list, t2


# KRONECKER
def kronecker(m, de, discrete = False):
    """
    Returns a discrete (+1/-1) or a Gaussian orthonormal Kronecker matrix
    by giving all the blocks for Kronecker product
    
    /!\ Kronecker matrix is not computed explicitly.
    
    Returns m blocks of dimension d2*de for one Kronecker matrix.
    """
    A_list = []
    
    if discrete:
        H = 1.0 / math.sqrt(2) * hadamard(de)
        
    for i in xrange(0, m):  
        if discrete: #+1/-1 entries
            # extended version of Hadamard matrix
            A = np.copy(H)
            # random multiplication *-1 on rows or columns
            # multiplication *-1 ?
            r = (np.random.rand(), np.random.randint(de), np.random.rand(), np.random.randint(de))
            if r[0] > 0.5:
                A[r[1], :] *= -1
            if r[2] > 0.5:
                A[:, r[3]] *= -1
        else:
            A = np.random.uniform(0.0, 1.0, de*de).reshape(de,de)
            A, _ = np.linalg.qr(A)
        A_list.append(A)  
    return A_list
 

# KRONECKER VECTOR (ONLY DISCRETE)

# auxiliary function
def rec_kronecker_vector(lev, height, l, beg, end, n_bits, b1, b2, M):
    """
    computes recursively a kronecker vector
    """
    if height == 0:
        #print('LEFT, lev, height', lev, height, beg, beg+l)
        #print(end)
        M[lev, beg: beg + l] = np.asarray([b1] * l)
        #print('RIGHT, lev, height', lev, height, end-l, end)
        #print(beg)
        M[lev, (end - l):end] = np.asarray([b2] * l)             
    else:
        if beg < l:
            assert l < end
            #print('end, l, end-l', end, l, end-l)
            rec_kronecker_vector(lev, height -1, l // n_bits, beg, l, n_bits, b1, b2, M)
            rec_kronecker_vector(lev, height -1, l // n_bits, l, end, n_bits, b1, b2, M)
        else:
            assert beg+l < end
            #print('beg, l, end, beg+l', beg, l, end, end-l)
            rec_kronecker_vector(lev, height -1, l // n_bits, beg, beg+l, n_bits, b1, b2, M)
            rec_kronecker_vector(lev, height -1, l // n_bits, beg+l, end, n_bits, b1, b2, M)           


def compute_kronecker_vector(d, n_lev, n_bits, discrete = True):
   
    """
    Return a Kronecker vector
   
    params:
    * n_lev : number of levels
    * n_bits : number of bits per level
   
    d is an integral power of 2
   
    """
    # if number of levels is inconsistant, returns a gaussian vector
    if d < n_bits**n_lev:
        return np.random.normal(0.0, 1.0, d)
       
    M = np.zeros((n_lev, d))
   
    for i in xrange(0, n_lev):
       
        if discrete:       
            # daw two random bits
            r  = np.random.rand()
            if r < 0.5:
                b1 = -1
            else:
                b1 = 1
            r  = np.random.rand()
            if r < 0.5:
                b2 = -1
            else:
                b2 = 1 
        else:
            b1 = np.random.normal(0.0, 1.0)          
            b2 = np.random.normal(0.0, 1.0)    

        rec_kronecker_vector(i, i, d // n_bits, 0, d, n_bits, b1, b2, M)
        #print('b1', b1)
        #print('b2', b2)
        #print(M[i])
        #print('')
    v = np.prod(M, axis = 0)
    del M
    #print(v.shape)
    #assert v.shape == (d,)
    return v   
    
    
    
    
    
    
#==============================================================================
# Matrix - vector products
#==============================================================================

# CIRCULANT PRODUCT
def circulant_product(x,y):
    """
    Returns product of a circulant matrix determined by its first column x
    and a vector y
    
    /!\ be careful, if rfft (instead of fft), it works only with even dimension
    """
    f = np.fft.rfft(y)
    g = np.fft.rfft(x)
    h = f * g
    z = np.fft.irfft(h)
    
#    f = np.fft.fft(y)
#    g = np.fft.fft(x)
#    h = f * g
#    z = np.fft.ifft(h)    
    return z



# SKEW-CIRCULANT PRODUCT
def skew_circulant_product(x, y, Omega, Omega_conj):
    """
    Returns product of a skew-circulant matrix determined by its first column x
    and a vector y
    
    /!\ be careful, dimension of x and y shoud be an integral power of 2
    """         
    x_ = x * Omega
    y_ = y * Omega
    fx_ = np.fft.fft(x_)
    fy_ = np.fft.fft(y_)
    
    fz = fx_ * fy_
    z = np.fft.ifft(fz) 
    z_ = z * Omega_conj
    return z_.real 



#TOEPLITZ PRODUCT
def toeplitz_product(c, r, x):
    """
    Returns product of a Toeplitz matrix determined by 
    its first column c + first row r
    and a vector x
    """        
    n = c.shape[0]
    s = np.zeros(n) # the first column of matrix S
    # /!\ don't need to construct the full matrix, only the first column
    # take row 0 from Tn, forget the first element and put  it as the first column of S
    s[1:] = r[1:][::-1] 
    #print('s', s)
    #print('r', r)
    c2n = np.zeros(2*n) # the first column of matrix C2n
    c2n[:n] = c
    c2n[n:] = s
    
    x_temp = np.zeros(2*n)
    x_temp[:n] = x
    
    y_temp = circulant_product(c2n,x_temp)
    y = y_temp[:n]
    return y
    
    
    
# HANKEL PRODUCT
def hankel_product(c, r, x):
    """
    Returns product of an Hankel matrix determined by 
    its first column c + last row r
    and a vector x
    """     
    c_ = c[::-1]
    return toeplitz_product(c_, r, x)[::-1]
        

# LOW DISPLACEMENT RANK PRODUCT
def low_displacement_product(G, H, r, x, Omega, Omega_conj):
    """
    Returns product of an Low-Displacement Rank matrix of Rank = r
    and a vector x.
    
    For better efficiency, compute in advance Omega and conjugate(Omega)
    (n-first square root of unity)
    
    M = sum^{i = r-1}_{i=0}  Circ[G[i]] * Skew[H[i]]
    
    H and G are d x r matrices with d dimension of x.
    Represent randomness budget for respectively Skew-circulant and Circulant matrices
    """
    #assert H.shape == (x.shape[0], r)
    #assert G.shape == (x.shape[0], r)
    y = np.zeros(x.shape[0])
    for i in xrange(r):
        y_ = skew_circulant_product(H[:, i], x, Omega, Omega_conj)       
        y += circulant_product(G[:, i], y_)
    return y


## HADAMARD
#def hadamard_transform(m, x): 
#    """
#    Returns product of an Hadamard matrix and a column vector x 
#    by using Hadamard transform.
#    
#    Parameter : m s.t. H is 2**m x 2**m and x column vector of dimension 2**m
#    
#    REMARK: H is normalized (factor 1/np.sqrt(2) ** m)
#    
#    """
#    print('DO NOT USE IT, IT IS NOT OPTIMIZED')
#    if m == 1:
#        H1 = np.ones((2,2))
#        H1[1,1] = -1
#        return np.dot(H1, x) * 1/(math.sqrt(2))
#    else:
#        res = np.zeros(2**m)
#        alpha = hadamard_transform(m-1, x[0:2**(m-1)]) 
#        beta = hadamard_transform(m-1, x[2**(m-1):])
#        res[0:2**(m-1)] = alpha + beta
#        res[2**(m-1):] = alpha - beta
#        return res * 1/(math.sqrt(2))


# HD3HD2HD1 product
def hadamard_cross_polytope_product(x, D, nblocks):
    """
    Returns product of prod_i HDi and column vector x
    
    /!\ x should be of a dimension which is an integral power of 2
    """
    # pseudo-random rotation HD_3HD_2H_1
    #dim = x.shape[0]
    #m = math.log(dim, 2)
    
    y = np.copy(x)
    for i in xrange(nblocks):
        #assert np.diag(D[i,:]).shape == (dim,dim)
        y = D[i,:] * y 
        #y = hadamard_transform(m,y)     
        #TODO: norm
        #y = fht.fht1(y)
        a = ffht.create_aligned(y.shape[0], np.float32)
        np.copyto(a, y)
        # TODO: CHOOSE A BETTER CHUNK SIZE
        ffht.fht(a, min(x.shape[0], 1024, 2048))   
        y = a
    return y
    
    
# KRONECKER PRODUCT: ITERATIVE VERSION
def kronecker_product(A_list, x):
    """
    OLD VERSION : slower from m = 11
    
    computes fast matrix-vector product for a Kronecker matrix
    
    A_list = [A_1, ..., A_M]: s.t
    
    K = A_1 x ... x A_M

    prerequisites:
    M is an integral power of 2 
    A_i supposed to be d_e x d_e orthonormal matrices    
    x is a d-dimensional column vector 
    d is an integral power of 2
    d = 2**M
    """
    m = len(A_list)
    #print('x.shape', x.shape)
    d = x.shape[0]
    #print('d', d, m)
    assert 2**m == d
    #assert A_list
    de = A_list[0].shape[0]
    #print('d', d)
    if m > 1:
        temp = np.dot( x.reshape(d/de, de, order = 'F'), A_list[0].T)
        kron = A_list[1]
        for i in xrange(2,m):
            kron = np.kron(kron, A_list[i])  
        y = np.dot( kron, temp)
        y = y.reshape(d, order = 'F')
    else:
        y = np.dot(A_list[0], x)
    return y
        
    
# KRONECKER PRODUCT: RECURSIVE VERSION
def kronecker_product_rec(A_list, x):
    """
    computes fast matrix-vector product for a Kronecker matrix
   
    A_list = [A_1, ..., A_M]: s.t
   
    K = A_1 x ... x A_M

    prerequisites:
    M is an integral power of 2
    A_i supposed to be d_e x d_e orthonormal matrices   
    x is a d-dimensional column vector
    d is an integral power of 2
    d = 2**M
    
    TO IMPLEMENT IN C FOR OPTIMIZATION
    """
    m = len(A_list)
    #print('x.shape', x.shape)
    d = x.shape[0]
    #assert A_list
    de = A_list[0].shape[0]
    #print('d', d, m)
    assert de**m == d
    #print('d', d)
    #temp = np.dot(x.reshape(d/de, de, order = 'F'), A_list[0].T)
    kron = rec_kronecker(A_list, de, d, 0, x) 
    #y = np.dot( kron, temp)
    return kron.reshape(d, order = 'F')


# auxiliary function for Kronecker product (recursive version)
def rec_kronecker(A_list, de, dim, j, x):
    """
    auxiliary recursive function to compute product of multiple kronecker product
    
    TO IMPLEMENT IN C FOR OPTIMIZATION
    """
   
    if dim == de:
        return np.dot(A_list[len(A_list)-1], x)
       
    result = np.empty(dim)
    #print('x.shape', x.shape, dim/de, de, j)
    new_x = np.dot( x.reshape(dim/de, de, order = 'F'), A_list[j].T)
    for i in xrange(0, de):
        result[(i*(dim/de)):((i+1)*(dim/de))] = rec_kronecker(A_list, de, (dim/de), (j+1), new_x[:,i])
    return result





#==============================================================================
# 
#
#
#                                  TESTS
#
#
#==============================================================================



#==============================================================================
# PRODUCT TESTS
#==============================================================================

def structure_product_tests(n_test, m, structure, nblocks = 3, r = 1, de = 2, discrete = True, n_K = 1):
    """
        Testing that matrix-vector product with structured matrices gives the same results
        with classical matrix-vector product.
    """
    
    if structure == 'circulant':        
        print "\n*** CIRCULANT MATRIX ***" # work only if an even value for n!!!!!
   
    elif structure == 'skew_circulant':
        print "\n*** SKEW CIRCULANT MATRIX ***"
        # completetely determined by its first coulumn 
        # depends on n paramters
        # upper triangle is negated
        #/!\ n should a be integral power of 2
    
    elif structure == 'hadamard':
        print "\n*** HADAMARD MATRIX ***"
        # we want the product Hx
    
    elif structure == 'hadamard_blocks':
        print "\n*** HADAMARD MATRIX + DIAGONAL MATRICES ***"  
        
    elif structure  == 'toeplitz':
        print "\n*** TOEPLITZ MATRIX ***"
        # completety determined by its first coulumn and first row
        # depends on 2n-1 parameters
    
    elif structure == 'hankel':
        print "\n*** HANKEL MATRIX ***"
        # completely determined by its first coulumn and last row
        # depends on 2n-1 parameters
    
    elif structure == 'low_displacement_rank':
        print "\n*** LOW DISPLACEMENT RANK R MATRIX ***"
    
    elif structure == 'kronecker':
        print "\n*** KRONECKER MATRICES ***" 
    
    n = 2**m
    assert r < n

    j = complex(0.0, 1.0)
    Omega = np.asarray([cmath.exp( j * cmath.pi * i / n) for i in xrange(n)])
    Omega_conj = np.conjugate(Omega)  
    
    t_better_avg = 0
    t_avg = 0
    

  
 
    for i in xrange(n_test): 


        if i % 10 == 0:
            print(i)

        x = np.random.normal(0.0, 1.0, n)
        # we want the product M x
        

        if structure == 'circulant':
            
            # x is the first column of the circulant matrix
            c = np.random.normal(0.0, 1.0, n)
            C = circulant(c)
            #toimage(C).show()
            
            t_better = time.time()
            y = circulant_product(c,x)
            print('\ncirculant')
            print(y)
            t_better = time.time() - t_better
            print 'execution time with fft ' + str(t_better)
              
            t = time.time()
            y = np.dot(C,x)
            print('\ndot')
            print(y)    
            t = time.time() - t
            print 'execution time with dot product ' + str(t)
            
     

        elif structure == 'skew_circulant':
            
            c = np.random.normal(0.0, 1.0, n)
            t = time.time()
            SkewC = skew_circulant(c)
            t = time.time() - t
            #print('t', t)
        
            t_better = time.time()
            print('\nskew circulant: faster if Omega and its conjugate are computed upfront')     
            y = skew_circulant_product(c, x, Omega, Omega_conj)
            print('y', y)    
            t_better = time.time() - t_better    
            print 'execution time with skew circulant product ' + str(t_better) 
            
            t = time.time()
            y = np.dot(SkewC,x) 
            print('\ndot')
            print(y)
            t = time.time() - t  
            print 'execution time with dot product ' + str(t)   
        
        elif structure == 'hadamard':
            
            t = time.time()    
            H = hadamard(2**m) * 1/((math.sqrt(2))**m)
            print('orthonormal?')
            print(np.dot(H, H.T))
            y = np.dot(H,x)
            print('\ndot')
            print(y)
            t = time.time() - t  
            print 'execution time with dot product ' + str(t)
            print('')
            
            print("hadamard transform from Barney")
            t_better = time.time()
            #y = hadamard_transform(m,x)
            y = fht.fht1(x)           
            print('\nhadamard')    
            print(y)
            t_better = time.time() - t_better
            print 'execution time with hadamard transform Barney version ' + str(t)  
            #print 'just a little bit faster because we do not need to precompute H'
            #assert np.array_equal(hadamard_transform(m,x), np.dot(H,x))
            #y_ = fht.fht.fht(x_)
            x_ = np.copy(x)
            #y = fht.fht1(y)
            t_c = time.time()
            #print(x_.shape[0])
            a = ffht.create_aligned(n, np.float32)
            np.copyto(a, x_)
            # TODO: CHOOSE A BETTER CHUNK SIZE
            ffht.fht(x_, min(x.shape[0], 1024, 2048))   
            y_ = x_
            t_c = time.time() - t_c  
            print('')
            print('FFHT  version')
            print(y_)
            print 'execution time with FFHT version ' + str(t_c)
            print('time we save with FFHT in comparison with Barbey', t_better/t_c)
            print ''
         
        elif structure == 'hadamard_blocks':
            
            D = compute_diagonal_matrices(2**m, nblocks)
            
            t = time.time() 
            H = hadamard(2**m)
            y = x
            for i in xrange(nblocks):
                y = np.dot(np.diag(D[i,:]), y)
                y = np.dot(H,y) 
            
            print('\ndot')
            print(y)
            t = time.time() - t  
            print 'execution time with dot product ' + str(t)
            
            print("hadamard transform")
            t_better = time.time()
            y = hadamard_cross_polytope_product(x, D, nblocks)  
            print('\nhadamard')    
            print(y)
            t_better = time.time() - t_better
            print 'execution time with hadamard transform ' + str(t)  
         
         
        elif structure == 'toeplitz':  
            
            c = np.random.normal(0.0, 1.0, n)
            r = np.random.normal(0.0, 1.0, n)
            r[0] = c[0]
            T = toeplitz(c,r)
            
            x = np.random.normal(0.0, 1.0, n)
            # we want the product Tnx
            
            t_better = time.time() 
            y = toeplitz_product(c, r, x)
            t_better = time.time() - t_better
            print('\ntoeplitz')
            print(y)    
            print 'execution time with toeplitz product ' + str(t_better)
            
            t = time.time()
            y = np.dot(T,x) 
            print('\ndot')
            print(y)
            t = time.time() - t 
            print 'execution time with dot product ' + str(t)

        elif structure == 'hankel':
            
            c = np.random.normal(0.0, 1.0, n)
            r = np.random.normal(0.0, 1.0, n)
            r[0] = c[n-1]
            Hank = hankel(c,r)
            
            t_better = time.time()
            print('\nhankel')
            y = hankel_product(c,r, x)
            print(y)
            t_better = time.time() - t_better
            print 'execution time with hankel product ' + str(t_better)
            
            t = time.time()
            y = np.dot(Hank,x) 
            print('\ndot')
            print(y)
            t = time.time() - t  
            print 'execution time with dot product ' + str(t)
        
        elif structure == 'low_displacement_rank':

            G = np.random.normal(0.0, 1.0, n*r).reshape(n,r)
            H = np.random.normal(0.0, 1.0, n*r).reshape(n,r)
 
            t = time.time()
            y = np.zeros(n)
            for j in xrange(r):
                 y_temp = np.dot(skew_circulant(H[:, j]), x)   
                 print(i, j, 'skew done')
                 y_temp = np.dot(circulant(G[:, j]), y_temp)
                 y += y_temp
                 print(i, 'circulant done')
                 
            print('\ndot')
            print('y', y)
            t = time.time() - t  
            print 'execution time with dot product ' + str(t) 
          
            t_better = time.time()
            print('\nlow displacement rank product: faster if dot product computes M during time')     
            y = low_displacement_product(G, H, r, x, Omega, Omega_conj)
            print('y', y)    
            t_better = time.time() - t_better   
            print 'execution time with low displacement rank product ' + str(t_better) 

        
        elif structure == 'kronecker':

            y_temp = np.copy(x)
            
            # sum of time for the blocks
            t_blocks = 0
            t_better_blocks = 0
    
            for j in xrange(n_K):
                
                # construct basic blocks for one Kronecker matrix
                A_list = []
    
                if discrete:
                    H = 1.0 / math.sqrt(2) * hadamard(de)
            
                for i in xrange(0, m):
                    
                    if discrete:
                        A = np.copy(H)
                        # random multiplication *-1 on rows or columns
                        # multiplication *-1 ?
                        r = (np.random.rand(), np.random.randint(de), np.random.rand(), np.random.randint(de))
                        if r[0] > 0.5:
                            A[r[1], :] *= -1
                        if r[2] > 0.5:
                            A[:, r[3]] *= -1
                    else:
                        A = np.random.normal(0.0, 1.0, 2*2).reshape(2,2)
                        A, _ = np.linalg.qr(A)
                    
                    A_list.append(A)  
        
            
                # construct the associate Kronecker matrix
                t = time.time()
                K = A_list[0]   
                if m > 1:
                    for i in xrange(1, m):
                        K = np.kron(K, A_list[i])
                #print('K.shape', K.shape)
                #assert K.shape == (d, d)    
                y = np.dot(K, y_temp)
                print('\ndot')
                print('y', y)
                #print('y.shape', y.shape)
                t = time.time() - t
                print 'execution time with dot product ' + str(t) 
            
        
                t_better = time.time()
                print('\nKronecker matrix')   
                y_rec = kronecker_product_rec(A_list, y_temp)
                print('y_rec', y_rec)  
                print('y.shape kronecker product', y_rec.shape)
                t_better = time.time() - t_better    
                print 'execution time with Kronecker product ' + str(t_better)  
                
                y_temp = y
                
                t_blocks += t
                t_better_blocks += t_better


        if structure != 'kronecker':
            t_better_avg += t_better
            t_avg += t
        else:
            t_better_avg += t_better_blocks
            t_avg += t_blocks
            
        print('t_avg', t_avg / n_test)
        print('t_better_avg', t_better_avg / n_test)
        print('time factor', t_avg, t_better_avg, t_avg/t_better_avg)  



#==============================================================================
# RANDOM MATRIX AGAINST STRUCTURED MATRICES
#==============================================================================
    
def structure_vs_random(n_test, m, structure, nblocks = 3, r = 1, de = 2, discrete = True, n_K = 1):
    
    if structure == 'circulant':        
        print "\n*** CIRCULANT MATRIX VS RANDOM MATRIX ***" # work only if an even value for n!!!!!
   
    elif structure == 'skew_circulant':
        print "\n*** SKEW CIRCULANT MATRIX VS RANDOM MATRIX ***"
        # completetely determined by its first coulumn 
        # depends on n paramters
        # upper triangle is negated
        #/!\ n should a be integral power of 2
    
    elif structure == 'dct':
        print "\n*** DISCRETE COSINE TRANSFORM VS RANDOM MATRIX ***"

    elif structure == 'hadamard':
        print "\n*** HADAMARD MATRIX VS RANDOM MATRIX ***"
        # we want the product Hx

    elif structure == 'hadamard_ffht':
        print "\n*** HADAMARD FFHT FROM FALCONN VS RANDOM MATRIX ***"
        
    elif structure == 'hadamard_fht':
        print "\n*** HADAMARD FHT FROM GITHUB VS RANDOM MATRIX ***"
 
    elif structure == 'hadamard_fwht_spiral':
        print "\n*** HADAMARD WHT FROM SPIRAL VS RANDOM MATRIX ***"
    
    elif structure == 'hadamard_fwht':
        print "\n*** HADAMARD FWHT VS RANDOM MATRIX ***"
     
    elif structure == 'hadamard_blocks':
        print "\n*** HADAMARD MATRIX + DIAGONAL MATRICES VS RANDOM MATRIX ***"  
        
    elif structure  == 'toeplitz':
        print "\n*** TOEPLITZ MATRIX VS RANDOM MATRIX ***"
        # completety determined by its first coulumn and first row
        # depends on 2n-1 parameters
    
    elif structure == 'hankel':
        print "\n*** HANKEL MATRIX VS RANDOM MATRIX ***"
        # completely determined by its first coulumn and last row
        # depends on 2n-1 parameters
    
    elif structure == 'low_displacement_rank':
        print "\n*** LOW DISPLACEMENT RANK R MATRIX VS RANDOM MATRIX ***"
    
    elif structure == 'kronecker':
        print "\n*** KRONECKER MATRICES VS RANDOM MATRIX ***" 
    
    n = 2**m
    assert r < n

    j = complex(0.0, 1.0)
    Omega = np.asarray([cmath.exp( j * cmath.pi * i / n) for i in xrange(n)])
    Omega_conj = np.conjugate(Omega)  
    
    # to measure time for matrix-vector product  while counting time to compute parameters
    t_better_avg = 0 
    t_rand_avg = 0
    
    # to measure time for matrix-vector product  without counting time to compute parameters
    t_prod1_avg = 0
    t_prod2_avg = 0
    

  
 
    for i in xrange(n_test): 


        if i % 20 == 0:
            print(i)

        x = np.random.normal(0.0, 1.0, n)
        # we want the product M x

        # comparison with random projection    
        t_rand = time.time()    
        A = np.random.normal(0.0, 1.0, n*n).reshape(n,n)
        t_prod1 = time.time() # to measure time without computing A
        y_rand = np.dot(A, x)
        t_prod1 = time.time() - t_prod1
        t_rand = time.time() - t_rand 
        #print('t_rand', t_rand)
        #print('t_prod1', t_prod1)
        

        if structure == 'circulant':
            
            t_better = time.time()
            # x is the first column of the circulant matrix
            c = np.random.normal(0.0, 1.0, n)
            t_prod2 = time.time()
            y = circulant_product(c,x)
            #print('\ncirculant')
            #print(z)
            t_prod2 = time.time() - t_prod2
            t_better = time.time() - t_better
            #print 'execution time with fft ' + str(t_better)          

        elif structure == 'skew_circulant':
            
            t_better = time.time()
            #print('\nskew circulant: faster if Omega and its conjugate are computed upfront')    
            c = np.random.normal(0.0, 1.0, n)
            t_prod2 = time.time()
            y = skew_circulant_product(c, x, Omega, Omega_conj)
            #print('y', y) 
            t_prod2 = time.time() - t_prod2
            t_better = time.time() - t_better    
            #print 'execution time with skew circulant product ' + str(t_better) 
        
        elif structure == 'dct':
            t_better = time.time()
            y = dct(x)
            t_better = time.time() - t_better
            t_prod2 = t_better

        elif structure == 'hadamard':
            
            #print("hadamard transform")
            t_better = time.time()
            y = hadamard_transform(m,x)
            #print('\nhadamard')    
            #print(y)
            t_better = time.time() - t_better
            t_prod2 = t_better
            #print 'execution time with hadamard transform ' + str(t)  
            #print 'just a little bit faster because we do not need to precompute H'
            #assert np.array_equal(hadamard_transform(m,x), np.dot(H,x))

        elif structure == 'hadamard_fht':
            t_better = time.time()
            y = fht.fht1(x)
            t_better = time.time() - t_better
            t_prod2 = t_better

        elif structure == 'hadamard_ffht':
            t_better = time.time()
            a = ffht.create_aligned(x.shape[0], np.float32)
            np.copyto(a, x)
            ffht.fht(a, min(x.shape[0], 1024, 2048))
            y = a
            t_better = time.time() - t_better
            t_prod2 = t_better
            
        elif structure == 'hadamard_fwht_spiral':
            t_better = time.time()
            y = fwht_spiral_copy(x)
            # fwht_spiral_inplace(x)
            t_better = time.time() - t_better
            t_prod2 = t_better

        elif structure == 'hadamard_fwht':
            t_better = time.time()
            y = fwht_copy(x)
            # fwht_inplace(x)
            t_better = time.time() - t_better
            t_prod2 = t_better
           
        elif structure == 'hadamard_blocks':
            
            #print("hadamard transform")
            t_better = time.time()
            D = compute_diagonal_matrices(2**m, nblocks)
            t_prod2 = time.time()
            y = hadamard_cross_polytope_product(x, D, nblocks)  
            #y = fht.fht1(x)
            t_prod2 = time.time() - t_prod2 
            #print('\nhadamard')    
            #print(y)
            t_better = time.time() - t_better
         
         
        elif structure == 'toeplitz':  
            
            t_better = time.time()
            c = np.random.normal(0.0, 1.0, n)
            r = np.random.normal(0.0, 1.0, n)
            r[0] = c[0]
            t_prod2 = time.time()
            y = toeplitz_product(c, r, x)
            t_prod2 = time.time() - t_prod2
            t_better = time.time() - t_better
            #print('\ntoeplitz')
            #print(y)    
            #print 'execution time with toeplitz product ' + str(t)

        elif structure == 'hankel':
            
            t_better = time.time()
            c = np.random.normal(0.0, 1.0, n)
            r = np.random.normal(0.0, 1.0, n)
            r[0] = c[n-1]
            t_prod2 = time.time()
            y = hankel_product(c,r, x)
            t_prod2 = time.time() - t_prod2
            t_better = time.time() - t_better
            #print 'execution time with hankel product ' + str(t)
        
        elif structure == 'low_displacement_rank':

            t_better = time.time()
            G = np.random.normal(0.0, 1.0, n*r).reshape(n,r)
            H = np.random.normal(0.0, 1.0, n*r).reshape(n,r)                
            #print('\nlow displacement rank product: faster if dot product computes M during time')  
            t_prod2 = time.time()
            y = low_displacement_product(G, H, r, x, Omega, Omega_conj)
            t_prod2 = time.time() - t_prod2 
            #print('y', y)    
            t_better = time.time() - t_better   
            #print 'execution time with low displacement rank product ' + str(t_better)
        
        elif structure == 'kronecker':
            # sum of time for the blocks
            t_better = time.time() 
            t_prod2 = 0
            #t_prod2 = time.time()
            y_temp = np.copy(x)
            for j in xrange(n_K):
                # construct basic blocks for one Kronecker matrix
                A_list = []
    
                if discrete:
                    H = 1.0 / math.sqrt(2) * hadamard(de)
            
                for i in xrange(0, m): 
                    if discrete:
                        A = np.copy(H)
                        # random multiplication *-1 on rows or columns
                        # multiplication *-1 ?
                        r = (np.random.rand(), np.random.randint(de), np.random.rand(), np.random.randint(de))
                        if r[0] > 0.5:
                            A[r[1], :] *= -1
                        if r[2] > 0.5:
                            A[:, r[3]] *= -1
                    else:
                        A = np.random.normal(0.0, 1.0, 2*2).reshape(2,2)
                        A, _ = np.linalg.qr(A)
                    
                    A_list.append(A)  
                t_prod2_part = time.time()
                y_temp = kronecker_product_rec(A_list, y_temp) 
                t_prod2_part = time.time() - t_prod2_part
                t_prod2 += t_prod2_part
            t_better = time.time() - t_better
            y = y_temp

        t_better_avg += t_better
        t_rand_avg += t_rand
        t_prod1_avg += t_prod1
        t_prod2_avg += t_prod2

    print('t_rand_avg', t_rand_avg / n_test)
    print('t_better_avg', t_better_avg / n_test)
    print('time factor while counting time to compute parameters', t_rand_avg/t_better_avg) 
    print('time factor without counting time to compute parameters', t_prod1_avg / n_test, t_prod2_avg/ n_test, t_prod1_avg / t_prod2_avg)
    



def threeStructBlocks_vs_random(n_test, m, B3, B2B1 , r = 1, de = 2, discrete = True):
    """
    """
    print "\n*** 3 Struct-Blocks VS RANDOM MATRIX ***"  
    print('B3', B3)

    d = de**m


    j = complex(0.0, 1.0)
    Omega = np.asarray([cmath.exp( j * cmath.pi * i / d) for i in xrange(d)])
    Omega_conj = np.conjugate(Omega)

    # to measure time for matrix-vector product  while counting time to compute parameters
    t_better_avg = 0
    t_rand_avg = 0

    # to measure time for matrix-vector product  without counting time to compute parameters
    t_prod1_avg = 0
    t_prod2_avg = 0    
        
    for i in xrange(n_test): 

        if i % 10 == 0:
            print(i)
            
        x = np.random.normal(0.0, 1.0, d)
        y_temp = np.copy(x)

        # comparison with random projection    
        t_rand = time.time()    
        A = np.random.normal(0.0, 1.0, d*d).reshape(d,d)
        t_prod1 = time.time() # to measure time without computing A
        z_rand = np.dot(A, x)
        t_prod1 = time.time() - t_prod1 
        t_rand = time.time() - t_rand  

        
        A_list = []

        t_block_B2B1 = time.time()
        
        if B2B1 == 'K2K1':
            n_K = 2 # number of Kronecker matrices
            # sum of time for the blocks
            t_B2B1 = time.time()
            for j in xrange(n_K):
                
                # construct basic blocks for one Kronecker matrix
                A_list = kronecker(m, de, discrete) 
                t_prod2 = time.time()
                y_temp = kronecker_product_rec(A_list, y_temp)
                t_prod2 = time.time() - t_prod2
                t_prod2_avg += t_prod2
                #print('y', y)        
                #print 'execution time with Kronecker product ' + str(t_better)             
            t_B2B1 = time.time() - t_B2B1
            t_better_avg += t_B2B1


        elif B2B1 == 'K2HD1':
            
            # block HD1
            time_first = time.time()
            D = compute_one_diagonal(d)
            t_prod2 = time.time()
            y_temp = D * y_temp
            #print(y_temp.shape)
            #assert y_temp.shape == (d,)
            #y_temp = hadamard_transform(m, y_temp) 
            #TODO
            a = ffht.create_aligned(y_temp.shape[0], np.float32)
            np.copyto(a, y_temp)
            ffht.fht(a, min(y_temp.shape[0], 1024, 2048))
            y_temp = a       
            #y_temp = fht.fht1(y_temp)            
            t_prod2 = time.time() - t_prod2
            time_first = time.time() - time_first
            t_better_avg += time_first
            t_prod2_avg += t_prod2
                    
    
            # block K2     
            time2 = time.time()
            A_list = kronecker(m, de, discrete) 
            t_prod2 = time.time()
            y_temp = kronecker_product_rec(A_list, y_temp)
            t_prod2 = time.time() - t_prod2
            time2 = time.time() - time2
            t_better_avg += time2
            t_prod2_avg += t_prod2

        elif B2B1 == 'HD2K1':
            
            # block K1     
            time2 = time.time()
            A_list = kronecker(m, de, discrete) 
            t_prod2 = time.time()
            y_temp = kronecker_product_rec(A_list, y_temp)
            t_prod2 = time.time() - t_prod2
            time2 = time.time() - time2
            t_better_avg += time2
            t_prod2_avg += t_prod2
            
            # block HD2
            time_first = time.time()
            D = compute_one_diagonal(d)
            t_prod2 = time.time()
            y_temp = D * y_temp
            #print(y_temp.shape)
            #assert y_temp.shape == (d,)
            #y_temp = hadamard_transform(m, y_temp)   
            #TODO
            a = ffht.create_aligned(y_temp.shape[0], np.float32)
            np.copyto(a, y_temp)
            ffht.fht(a, min(y_temp.shape[0], 1024, 2048))
            y_temp = a       
            #y_temp = fht.fht1(y_temp)
            t_prod2 = time.time() - t_prod2
            time_first = time.time() - time_first
            t_better_avg += time_first
            t_prod2_avg += t_prod2            
            
        elif B2B1 == 'HD2HD1':
            # block HD2 HD1
            t_B2B1 = time.time()
            D = compute_diagonal_matrices(d, nblocks = 2)
            t_prod2 = time.time()
            y_temp = hadamard_cross_polytope_product(y_temp, D, nblocks = 2)    
            t_prod2 = time.time() - t_prod2
            t_B2B1 = time.time() - t_B2B1
            t_better_avg += t_B2B1
            t_prod2_avg += t_prod2 
            
        elif B2B1 == 'D2K1':

            # block K1     
            time1 = time.time()
            A_list = kronecker(m, de, discrete) 
            t_prod2 = time.time()
            y_temp = kronecker_product_rec(A_list, y_temp)
            t_prod2 = time.time() - t_prod2
            time1 = time.time() - time1
            t_better_avg += time1
            t_prod2_avg += t_prod2 
            
            # block D2   
            time2 = time.time()
            D = compute_one_diagonal(d)
            t_prod2 = time.time()
            y_temp = D * y_temp
            t_prod2 = time.time()
            time2 = time.time() - time2
            t_better_avg += time2
            t_prod2_avg += t_prod2
        
        elif B2B1 == 'D2HD1':
            # block HD1
            time_first = time.time()
            D = compute_one_diagonal(d)
            t_prod2 = time.time()
            y_temp = D * y_temp
            #print(y_temp.shape)
            #assert y_temp.shape == (d,)
            #y_temp = hadamard_transform(m, y_temp)   
            #TODO
            a = ffht.create_aligned(y_temp.shape[0], np.float32)
            np.copyto(a, y_temp)
            ffht.fht(a, min(y_temp.shape[0], 1024, 2048))
            y_temp = a       
            #y_temp = fht.fht1(y_temp)
            t_prod2 = time.time() - t_prod2 
            time_first = time.time() - time_first
            t_better_avg += time_first
            t_prod2_avg += t_prod2
            
            # block D2   
            time2 = time.time()
            D = compute_one_diagonal(d)
            t_prod2 = time.time()
            y_temp = D * y_temp
            t_prod2 = time.time() - t_prod2 
            time2 = time.time() - time2
            t_better_avg += time2  
            t_prod2_avg += t_prod2
        
        else:
            print('ERROR B2B1')

        t_block_B2B1 = time.time() - t_block_B2B1
        # BLOCK 3
    
        t_B3 = time.time() 
        
        if B3 == 'circulant':
            # apply last block 3 : circulant block        
            t_circ = time.time()
            # c is the first column of the circulant matrix
            c = np.random.normal(0.0, 1.0, d)
            t_prod2 = time.time()
            y_temp = circulant_product(c,y_temp)
            t_prod2 = time.time() - t_prod2 
            t_circ = time.time() - t_circ   
            t_better_avg += t_circ 
            t_prod2_avg += t_prod2
            
        elif B3 == 'skew_circulant':

            # apply last block 3 : skew-circulant block        
            t_struc = time.time()
            # c is the first column of the skew-circulant matrix
            c = np.random.normal(0.0, 1.0, d)
            t_prod2 = time.time()
            y_temp = skew_circulant_product(c, y_temp, Omega, Omega_conj)
            t_prod2 = time.time() - t_prod2
            t_struc = time.time() - t_struc   
            t_better_avg += t_struc 
            t_prod2_avg += t_prod2
        
        elif B3 == 'toeplitz':

            # apply last block 3 : toeplitz block        
            t_struc = time.time()
            # c is the first column of the toeplitz matrix
            c = np.random.normal(0.0, 1.0, d)
            # r is first row
            r = np.random.normal(0.0, 1.0, d)
            r[0] = c[0] 
            t_prod2 = time.time()
            y_temp = toeplitz_product(c, r, y_temp)   
            t_prod2 = time.time() - t_prod2
            t_struc = time.time() - t_struc   
            t_better_avg += t_struc 
            t_prod2_avg += t_prod2

        elif B3 == 'hankel':
            
            # apply last block 3 : hankel block      
            t_struc = time.time()
            # c is the first column 
            c = np.random.normal(0.0, 1.0, d)
            # r is last row
            r = np.random.normal(0.0, 1.0, d)
            r[0] = c[n-1] 
            t_prod2 = time.time()
            y_temp = hankel_product(c, r, y_temp)  
            t_prod2 = time.time() - t_prod2 
            t_struc = time.time() - t_struc   
            t_better_avg += t_struc 
            t_prod2_avg += t_prod2
            
        elif B3 == 'low_displacement_rank':
            
            # apply last block 3 : low displacement rank block       
            t_struc = time.time()
            G = np.random.normal(0.0, 1.0, d*r).reshape(d,r)
            H = np.random.normal(0.0, 1.0, d*r).reshape(d,r)  
            t_prod2 = time.time()              
            y_temp = low_displacement_product(G, H, r, y_temp, Omega, Omega_conj)    
            t_prod2 = time.time() - t_prod2
            t_struc = time.time() - t_struc   
            t_better_avg += t_struc 
            t_prod2_avg += t_prod2
            
        elif B3 == 'H_kronecker_vector':

            # apply last L1L2 block
            # L1 L2 = H Diag(Kronecker vector) 
            time_last = time.time()
            # L2 = Kronecker vector
            vector = compute_kronecker_vector(d, n_lev = 3, n_bits = de, discrete = True)     
            t_prod2 = time.time()
            y_temp = vector * y_temp
            #print(y_temp.shape)
            #assert y_temp.shape == (d,)
            # L1 = H
            #y_temp = hadamard_transform(m, y_temp)
            #TODO
            a = ffht.create_aligned(y_temp.shape[0], np.float32)
            np.copyto(a, y_temp)
            ffht.fht(a, min(y_temp.shape[0], 1024, 2048))
            y_temp = a       
            #TODO
            #y_temp = fht.fht1(y_temp)
            t_prod2 = time.time() - t_prod2
            #assert y_temp.shape == (d,)            
            time_last = time.time() - time_last
            t_better_avg += time_last
            t_prod2_avg += t_prod2
            
        elif B3 == 'K_kronecker_vector':
            # apply last L1L2 block
            # L1 L2 = K Diag(Kronecker vector) 
            time_last = time.time()
            # L2 = Kronecker vector
            vector = compute_kronecker_vector(d, n_lev = 3, n_bits = de, discrete = True)
            A_list2 = kronecker(m, de, discrete)
            t_prod2 = time.time()
            y_temp = vector * y_temp
            #print(y_temp.shape)
            #assert y_temp.shape == (d,)            
            # L1 = K
            y_temp = kronecker_product_rec(A_list2, y_temp)
            t_prod2 = time.time() - t_prod2
            #assert y_temp.shape == (d,)
            
            time_last = time.time() - time_last
            t_better_avg += time_last
            t_prod2_avg += t_prod2
        
        elif B3 == 'hadamard':
            t_better = time.time()
            D = compute_one_diagonal(d)
            t_prod2 = time.time()
            y_temp = D * y_temp
            
            #TODO
            a = ffht.create_aligned(y_temp.shape[0], np.float32)
            np.copyto(a, y_temp)
            ffht.fht(a, min(y_temp.shape[0], 1024, 2048))
            y_temp = a       
            #TODO
            #y_temp = fht.fht1(y_temp)
            
            t_prod2 = time.time() - t_prod2
            t_better = time.time() - t_better
            t_prod2_avg += t_prod2
            t_better_avg += t_better
        else:
            print('ERROR B3')
        
        t_B3 = time.time() - t_B3
        
        #print('t_block_B2B1', t_block_B2B1)
        #print('t_B3', t_B3)
        
            
        t_rand_avg += t_rand
        t_prod1_avg += t_prod1
     
    print('t_rand_avg', t_rand_avg / n_test)
    print('t_better_avg', t_better_avg / n_test)
    factor = t_rand_avg/t_better_avg
    print('time factor while counting parameters computation', factor)    
    print('time factor without counting time to compute parameters', t_prod1_avg / n_test, t_prod2_avg/ n_test, t_prod1_avg / t_prod2_avg)
    return t_prod1_avg / t_prod2_avg
    
if __name__ == '__main__':  

    #OMP_NUM_THREADS=1 python StructuredMatrices.py  
    
    n_test = 1
 
    # data dimension for all cases
    #n = 10000
    #n = 10000
    m = 9
    m = 5
    n = 2**m

#==============================================================================
# TESTS CHECKING THAT MATRIX-VECTOR PRODUCT RESULT IS RIGHT.
#==============================================================================
    print('*** PRODUCT TESTS ***')
    
#    structure_product_tests(n_test, m, 'circulant')
#    structure_product_tests(n_test, m, 'skew_circulant')
#    structure_product_tests(n_test, m, 'hadamard')
#    structure_product_tests(n_test, m, 'hadamard_blocks', nblocks = 3)
#    structure_product_tests(n_test, m, 'toeplitz')
#    structure_product_tests(n_test, m, 'hankel')
#    structure_product_tests(n_test, m, 'low_displacement_rank', r = 1)
    structure_product_tests(n_test, m, 'kronecker', de = 2, discrete = True, n_K = 1)


#==============================================================================
# TESTS CHECKING SPEED-UPS FOR MATRIX-VECTOR PRODUCT WITH STRUCTURED MATRICES
#==============================================================================
   
    print('')
    print('*** RANDOM MATRIX AGAINST STRUCTURED MATRICES ***')
    
#    structure_vs_random(n_test, m, 'dct')
#    structure_vs_random(n_test, m, 'hadamard') # from m = 7
#    structure_vs_random(n_test, m, 'hadamard_fht')
#    structure_vs_random(n_test, m, 'hadamard_ffht')    
#    structure_vs_random(n_test, m, 'hadamard_fwht_spiral')
#    structure_vs_random(n_test, m, 'hadamard_fwht') # from m = 7
#    # sys.exit(0)
#    structure_vs_random(n_test, m, 'hadamard_blocks', nblocks = 3) # from m = 10

#    structure_vs_random(n_test, m, 'circulant') # better from m = 6
#    structure_vs_random(n_test, m, 'skew_circulant') # better from m = 6
#    structure_vs_random(n_test, m, 'toeplitz') # better from m = 6
#    structure_vs_random(n_test, m, 'hankel') # better from m = 6
#    structure_vs_random(n_test, m, 'low_displacement_rank', r = 1) # better from m = 6 
#    structure_vs_random(n_test, m, 'low_displacement_rank', r = 2) # better from m = 7 
#    structure_vs_random(n_test, m, 'low_displacement_rank', r = 3) # better from m = 7 
#    structure_vs_random(n_test, m, 'low_displacement_rank', r = 5) # better from m = 7 
#    structure_vs_random(n_test, m, 'low_displacement_rank', r = 10) # better from m = 8 
#    structure_vs_random(n_test, m, 'low_displacement_rank', r = 50) # better from m = 9  
#    structure_vs_random(n_test, m, 'low_displacement_rank', r = 100) # better from m = 10     
    structure_vs_random(n_test, m, 'kronecker', de = 2, discrete = True, n_K = 1) # better from m = 8 


    print('')
    print('*** RANDOM MATRIX AGAINST 3-Struct-Block ***')
    
    # *** B3 K2 K1 ***
#    print('B3 K2 K1')      
#    threeStructBlocks_vs_random(n_test, m, 'circulant', 'K2K1')
#    threeStructBlocks_vs_random(n_test, m, 'skew_circulant', 'K2K1')
#    threeStructBlocks_vs_random(n_test, m, 'toeplitz', 'K2K1')
#    threeStructBlocks_vs_random(n_test, m, 'hankel', 'K2K1')
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'K2K1', r = 1)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'K2K1', r = 2)    
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'K2K1', r = 3)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'K2K1', r = 5)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'K2K1', r = 10)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'K2K1', r = 50)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'K2K1', r = 100)
#    threeStructBlocks_vs_random(n_test, m, 'H_kronecker_vector', 'K2K1')
#    threeStructBlocks_vs_random(n_test, m, 'K_kronecker_vector', 'K2K1')
    
    # *** B3 K2 HD1 *** 
#    print('B3 K2 HD1')      
#    threeStructBlocks_vs_random(n_test, m, 'circulant', 'K2HD1')
#    threeStructBlocks_vs_random(n_test, m, 'H_kronecker_vector', 'K2HD1')
#    threeStructBlocks_vs_random(n_test, m, 'K_kronecker_vector', 'K2HD1')    
#    threeStructBlocks_vs_random(n_test, m, 'skew_circulant', 'K2HD1')
#    threeStructBlocks_vs_random(n_test, m, 'toeplitz', 'K2HD1')
#    threeStructBlocks_vs_random(n_test, m, 'hankel', 'K2HD1')
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'K2HD1', r = 1)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'K2HD1', r = 2)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'K2HD1', r = 3)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'K2HD1', r = 5)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'K2HD1', r = 10)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'K2HD1', r = 50)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'K2HD1', r = 100)

   
    # *** B3 HD2 K1 ***
#    print('B3 HD2 K1')    
#    threeStructBlocks_vs_random(n_test, m, 'circulant', 'HD2K1')
#    threeStructBlocks_vs_random(n_test, m, 'skew_circulant', 'HD2K1')
#    threeStructBlocks_vs_random(n_test, m, 'toeplitz', 'HD2K1')
#    threeStructBlocks_vs_random(n_test, m, 'hankel', 'HD2K1')
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'HD2K1', r = 1)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'HD2K1', r = 2)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'HD2K1', r = 3)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'HD2K1', r = 5)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'HD2K1', r = 10)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'HD2K1', r = 50)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'HD2K1', r = 100)    
#    threeStructBlocks_vs_random(n_test, m, 'H_kronecker_vector', 'HD2K1')
#    threeStructBlocks_vs_random(n_test, m, 'K_kronecker_vector', 'HD2K1')   

    # *** B3 HD2 HD1 ***
#    print('B3 HD2 HD1') 
#    threeStructBlocks_vs_random(n_test, m, 'hadamard', 'HD2HD1')
#    threeStructBlocks_vs_random(n_test, m, 'circulant', 'HD2HD1')
#    threeStructBlocks_vs_random(n_test, m, 'H_kronecker_vector', 'HD2HD1')
##    threeStructBlocks_vs_random(n_test, m, 'K_kronecker_vector', 'HD2HD1') 
#    threeStructBlocks_vs_random(n_test, m, 'skew_circulant', 'HD2HD1')
#    threeStructBlocks_vs_random(n_test, m, 'toeplitz', 'HD2HD1')
#    threeStructBlocks_vs_random(n_test, m, 'hankel', 'HD2HD1')
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'HD2HD1', r = 1)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'HD2HD1', r = 2)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'HD2HD1', r = 3)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'HD2HD1', r = 5)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'HD2HD1', r = 10)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'HD2HD1', r = 50)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'HD2HD1', r = 100)   

    # *** B3 D2 K1 ***
#    print('B3 D2 K1')
#    threeStructBlocks_vs_random(n_test, m, 'circulant', 'D2K1')
#    threeStructBlocks_vs_random(n_test, m, 'H_kronecker_vector', 'D2K1')
#    threeStructBlocks_vs_random(n_test, m, 'K_kronecker_vector', 'D2K1')
#    threeStructBlocks_vs_random(n_test, m, 'skew_circulant', 'D2K1')
#    threeStructBlocks_vs_random(n_test, m, 'toeplitz', 'D2K1')
#    threeStructBlocks_vs_random(n_test, m, 'hankel', 'D2K1')
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'D2K1', r = 1)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'D2K1', r = 2)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'D2K1', r = 3)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'D2K1', r = 5)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'D2K1', r = 10)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'D2K1', r = 50)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'D2K1', r = 100)


    # *** B3 D2 HD1 ***  
#    print('B3 D2 HD1') 
#    threeStructBlocks_vs_random(n_test, m, 'circulant', 'D2HD1')
#    threeStructBlocks_vs_random(n_test, m, 'H_kronecker_vector', 'D2HD1')
#    threeStructBlocks_vs_random(n_test, m, 'K_kronecker_vector', 'D2HD1')
#    threeStructBlocks_vs_random(n_test, m, 'skew_circulant', 'D2HD1')
#    threeStructBlocks_vs_random(n_test, m, 'toeplitz', 'D2HD1')
#    threeStructBlocks_vs_random(n_test, m, 'hankel', 'D2HD1')
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'D2HD1', r = 1)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'D2HD1', r = 2)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'D2HD1', r = 3)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'D2HD1', r = 5)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'D2HD1', r = 10)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'D2HD1', r = 50)
#    threeStructBlocks_vs_random(n_test, m, 'low_displacement_rank', 'D2HD1', r = 100)

    
    
#    print('')
#    print('*** RANDOM MATRIX AGAINST 3-Struct-Block : PRINT CURVES ***')    
#    #x = [2**9, 2**10, 2**11, 2**12]
#    x = [9, 10, 11, 12]
#    
#    # HD1 HD2 HD3
#    y1 = [0.9176520842962655, 1.9260485858743994, 3.751413299304557, 5.705348678466687]
#    
#    # best results: S-Circ D2 HD1
#    y2 = [5.506644298458103, 9.367204580058601, 20.258289321931354, 46.74141325570161]
#    
#    # worst results: Rank100 K2 K1
#    y3 = [0.5501480562340193, 1.1753992390370407, 2.208282103251001, 4.729328053355894]
#    
#    # H Diag(Kronecker) D2 HD1
#    y4 = [3.129590745119075, 6.584041691543052, 11.926523817606581, 25.67217856626332]
#    
#    # H Diag(Kronecker) D2 K1
#    y5 = [1.8498619539111747, 3.3464723072641247, 8.184942603671706, 15.024057306282089]
#    
#    # K Diag(Kronecker) D2 HD1
#    y6 = [1.7273614576138687, 3.836925311850889, 7.398289413038173, 15.439655759313482]
#    
#    # K Diag(Kronecker) D2 K1
#    y7 = [1.2567947216799678, 2.4578874166755695, 4.938414211825468, 10.46717622633485]
#    
#    # Rank5 D2 K1
#    y8 = [2.004678424970993, 4.621614382124385, 9.53647804658827, 17.996377678268058]
#    
#    
## http://www.labri.fr/perso/nrougier/teaching/matplotlib/
## http://matplotlib.org/examples/color/named_colors.html    
#    plt.plot(x, y1, '-', color = 'darkorange', label = 'HD3 HD2 HD1', linewidth=1.5)
#    plt.plot(x, y2, '-', color = 'tomato', label = 'SCirc D2 HD1', linewidth=1.5)
#    plt.plot(x, y3, '-', color = 'seagreen', label = 'R100 K2 K1', linewidth=1.5)
#    plt.plot(x, y4, '-', color = 'mediumpurple', label = 'H Diag(K) D2 HD1', linewidth=1.5)
#    plt.plot(x, y5, '-', color = 'magenta', label = 'H Diag(K) D2 K1', linewidth=1.5)
#    plt.plot(x, y6, '-', color = 'yellow', label = 'K Diag(K) D2 HD1', linewidth=1.5)
#    plt.plot(x, y7, '-', color = 'deepskyblue', label = 'K Diag(K) D2 K1', linewidth=1.5)
#    plt.plot(x, y8, '-', color = '#03ED3A', label = 'R5 D2 K1', linewidth=1.5)
#    plt.plot(x, [1, 1, 1, 1], '-', label = 'R5 D2 K1', linewidth=1.5)
#    
#    plt.xticks([9, 10, 11, 12],
#       [r'$2^{9}$', r'$2^{10}$', r'$2^{11}$', r'$2^{12}$'], fontsize = 20)
#    plt.xlabel('Data dimension', fontsize = 15)
#    plt.ylabel('Factor speed-up', fontsize = 15)
#    plt.xlim(9, 12)
#    plt.title('Speed-ups computation for matrix-vector product', fontsize = 20)
#    plt.legend(loc='upper left', frameon=False)
#    plt.grid()    
#    #http://matplotlib.org/1.3.1/api/pyplot_api.html#matplotlib.pyplot.legend
##    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=9,
##           ncol=7, mode="expand", borderaxespad=0.)
#    plt.show()    
#    










    
    


   

    
    
  
  
