# -*- coding: utf-8 -*-
"""
Created on Fri May 13 21:40:54 2016

@author: amorvan
"""
from __future__ import division
import StructuredMatrices
import math, cmath
import numpy as np
import time
import ffht # Fast Fast Hadamard Transform from FALCONN project
import scipy.spatial.distance
#import matplotlib.pyplot as plt



#==============================================================================
#  Basic functions
#==============================================================================


def unpickle(file):
    """ primitive to unpickle a file """
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict




def pickle(obj, filename):
    """ primitive to pickle a file """
    # https://github.com/numpy/numpy/issues/2396
    # it fails sometimes for big arrays
    import cPickle
    f = file(filename, 'wb')
    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    
    
    
#==============================================================================
# Hash functions class
#==============================================================================

#def cross_polytope_hashing(x):
#    """
#    Find the closest neighbor to x from basis {+-e_i}
#    => find the largest coefficient
#    
#    x is supposed to be normalized
#    
#    here we return all the indices of absolute largest coefficient
#    because for multi-probing it is usefull.
#    
#    But for this application here, it is better to use the next function
#    returning only THE nearest canonical basis vector
#    """
#    idx = np.argsort(np.abs(x), axis = None)
#    #TODO: if it a bottleneck to reverse...
#    return idx[::-1]

def cross_polytope_hashing_basic(x):
    """
    Find the closest neighbor to x from basis {+-e_i}
    => find the largest coefficient
    
    x is supposed to be normalized
    """
    return np.argmax(np.abs(x))


def sign(x):
    """
    sign function returning 0 or 1
    """
    x_bin = np.zeros(x.shape, dtype = 'int')
    x_bin[x >= 0] = 1
    return x_bin    

def sign11(x):
    """
    sign function returning -1 or 1
    """    
    x_bin = np.ones(x.shape, dtype = 'int')
    x_bin[x < 0] = -1
    return x_bin   
    
    
#==============================================================================
# Block class    
#==============================================================================

class Block(object):

    """
    A = B3 B2 B1
    
    Each matrix Bi is represented with corresponding information by a Block class object
    
    Parameters:
    * d : data dimension
    * hash_func : name of the structured matrix for this block
    
    """
    
    def __init__(self, d, hash_func, bonus):

        # intialization

        self._d = d  
        self._hash_func = hash_func
        
        self._A = None
        
        self._col = None
        self._row = None    

        # specific to an HDi block
        self._nblocks = None
        self._D = None

        # specific to a low-displacement rank block
        self._r = None
        self._G = None
        self._H = None
        
        # specific to a kronecker block
        self._discrete = None # boolean
        self._de = None
        self._A_list = None
        
        # to be sure I forgot no cases, to be sure we enter at least in one if
        self._check = False    
 
      
        # all available structured block
        assert self._hash_func in ['random', 'circulant', 'skew_circulant', 'toeplitz', 'hankel', 'pure_hadamard', 'hadamard', 'low_displacement_rank', 'kronecker', 'diagonal', 'diagonal_kronecker', 'diagonal_gaussian' ]

        # filling necessary variables
        
        if self._hash_func == 'random':
            self._A = np.random.normal(0.0, 1.0, (self._d, self._d))
            self._check = True
            
        if self._hash_func == 'circulant' or hash_func == 'skew_circulant' or hash_func == 'toeplitz' or hash_func == 'hankel':
            self._col = np.random.normal(0.0, 1.0, self._d)
            self._check = True
           
        if self._hash_func == 'hadamard':
            assert 'nblocks' in bonus
            self._nblocks = bonus['nblocks']
            self._D = StructuredMatrices.compute_diagonal_matrices(self._d, self._nblocks)
            self._check = True
            # do not compute Hadamard matrix in advance
            
        if self._hash_func == 'toeplitz':
            self._row = np.random.normal(0.0, 1.0, self._d)
            self._row[0] = self._col[0]
            self._check = True
            
        if self._hash_func == 'hankel':
            self._row = np.random.normal(0.0, 1.0, self._d)
            self._row[0] = self._col[self._d - 1] 
            self._check = True
        
        if self._hash_func == 'low_displacement_rank':
            assert 'r' in bonus
            self._r = bonus['r']
            self._G = np.random.normal(0.0, 1.0, (self._d, self._r))
            self._H = np.random.normal(0.0, 1.0, (self._d, self._r))  
            self._check = True
                
        if self._hash_func == 'kronecker':
            assert 'discrete' in bonus 
            assert 'de' in bonus
            self._discrete = bonus['discrete']
            self._de = bonus['de']  
            self._A_list = StructuredMatrices.kronecker( int(math.log(self._d, 2)), self._de, self._discrete)
            self._check = True
            
        if self._hash_func == 'diagonal':
            self._D = StructuredMatrices.compute_diagonal_matrices(self._d, 1).reshape(self._d,)
            self._check = True
        
        if self._hash_func == 'diagonal_kronecker':
            self._D = StructuredMatrices.compute_kronecker_vector(self._d, n_lev = 3, n_bits = 2, discrete = True)
            self._check = True
            
        if self._hash_func == 'diagonal_gaussian':
            self._D = np.random.normal(0.0, 1.0, self._d)  
            self._check = True
            
        if self._hash_func == 'pure_hadamard':
            self._check = True
            # no parameters
            pass
     
        assert self._check
     
     
     
#==============================================================================
#  Hash function class    
#==============================================================================
     
     
class HashFunction(object):
    
    def __init__(self, k_hash, d, d_pr, kernel, hash_func_list, bonus):  
        
        self._k_hash = k_hash # number of hash functions to collide
        self._d = d  
        self._d_pr = d_pr
        self._kernel = kernel # /!\  only Cross-polytope has been tested
        self._hash_func_list = hash_func_list
          
        # information corresponding to 3 different blocks (for each k_hash functions)
        self._BBB = [       [Block(self._d , self._hash_func_list[i], bonus[i]) for i in xrange(len(hash_func_list))]          for j in xrange(self._k_hash)  ]    
 

    def general_one_block_hashing(self, i, x, OMEGA, OMEGA_CONJ):
        
            """
                returns one hash value corresponding to 3 blocks
            """
            
            y = np.copy(x)
            
            t_for = time.time()
            t_sum = 0            
            # TODO: we can not avoid the for loop because each block is not independent
            for block in self._BBB[i][::-1]:
    
                # matrix-vector product
    
                t_rand = time.time()
                if block._hash_func == 'random':
                    y = np.dot(block._A, y) 
                t_rand = time.time() - t_rand
                   
                t_circ = time.time()
                if block._hash_func == 'circulant': 
                    y = StructuredMatrices.circulant_product(block._col, y)
                t_circ = time.time() - t_circ
                t_skew = time.time()
                if block._hash_func == 'skew_circulant':
                    y = StructuredMatrices.skew_circulant_product(block._col , y, OMEGA, OMEGA_CONJ)
                t_skew = time.time() - t_skew
                
                t_toep = time.time()
                if block._hash_func == 'toeplitz':
                    y = StructuredMatrices.toeplitz_product(block._col, block._row, y)
                t_toep = time.time() - t_toep
                
                t_had = time.time()            
                if block._hash_func == 'hadamard':
                    y = StructuredMatrices.hadamard_cross_polytope_product(y, block._D, block._nblocks)
                t_had = time.time() - t_had
                
                t_hank = time.time()                
                if block._hash_func == 'hankel':
                    y = StructuredMatrices.hankel_product(block._col, block._row, y)
                t_hank = time.time() - t_hank
                
                t_rank = time.time()            
                if block._hash_func == 'low_displacement_rank':        
                    y = StructuredMatrices.low_displacement_product(block._G, block._H, block._r, y, OMEGA, OMEGA_CONJ)
                t_rank = time.time() - t_rank
                
                t_kro = time.time()
                if block._hash_func == 'kronecker': 
                    y = StructuredMatrices.kronecker_product_rec(block._A_list, y)
                t_kro = time.time() - t_kro
                
                t_h = time.time()                   
                if block._hash_func == 'pure_hadamard':
                    #TODO: normalization
                    a = ffht.create_aligned(y.shape[0], np.float32)
                    np.copyto(a, y)
                    ffht.fht(a, min(x.shape[0], 1024, 2048))   
                    y = a                    
                    
                t_h = time.time() - t_h
    
                t_diag = time.time()
                if block._hash_func == 'diagonal' or block._hash_func == 'diagonal_kronecker' or block._hash_func == 'diagonal_gaussian':
                    y = block._D * y
                t_diag = time.time() - t_diag
                
                t_sum += t_rand + t_circ + t_skew + t_toep + t_had + t_hank + t_rank + t_kro + t_h + t_diag
                  
                #print('block', block._hash_func, t_rand, t_circ, t_skew, t_toep, t_had, t_hank, t_rank, t_kro, t_h, t_diag)
                #print(block, y.shape)
            t_for = time.time() - t_for
            #print('t_for', t_for)
            #print('t_sum', t_sum)
    
            # all blocks have been handled.
                
            # dimensionality reduction
            y = y[:self._d_pr]    
            
            # normalization
            y /= np.linalg.norm(y)
            
            # cross_polytope_hashing
            if self._kernel == cross_polytope_hashing_basic:
                h = self._kernel(y) 
                
                if y[h] < 0:
                    h += x.shape[0] # d
                #print('h', h, y)
                    
                # /!\ this way we are sure to have a bijective function
                # we can not say h = h1h2... because we do not return 1 bit 
                #but for each hash function a number betwen 0 and 2d-1
                return h * (1 + 2 * self._d * i) 

            else:
                print('ERROR, no other option for kernel function')
#            else:
#                #TODO: I did not work on this kernel
#                print('not working')
#                bits = self._kernel(y)
#                #TODO: for the moment returns a string key to improve with bits operation 
#                # http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.packbits.html
#                # TO TEST
#                # https://wiki.python.org/moin/PythonSpeed/PerformanceTips
#                b_list = [str(bit) for bit in bits]
#                b = "".join(b_list)
#                return b            


                
    def general_hashing(self, x, OMEGA, OMEGA_CONJ):
        """
         collide k_hash functions to have the complete hash key  
         """         
         
        h_k_hash_list = [ self.general_one_block_hashing(i, x, OMEGA, OMEGA_CONJ) for i in range(self._k_hash)]
#TODO:
        complete_hash = sum(h_k_hash_list)       
        return complete_hash



#==============================================================================
# Functions relatively to collision probability collision
#==============================================================================



def random_points_with_distance_in(a, b, d):
    """
    returns two random points of dimension d
    with a, b corresponding to bounds of the considered interval for euclidian distance
    between with those two points
    """

    # First point
    x = np.random.uniform(-1, 1, d) # faster than to use normal version   
    # Numerical precaution ; not a problem for randomness as the corresponding 
    # points on the sphere appear with proba zero
    while (x[0] == 0):
        x[0] = np.random.uniform(-1, 1, 1)
    x = x/np.linalg.norm(x)

    # Second point
    z = np.random.uniform(-1, 1, d)
    # we want z to be part of the hyperplan touching the sphere and x
    z[0] = - np.dot(np.delete(x, 0), np.delete(z, 0))/x[0]
    z = z/np.linalg.norm(z)
    #print('orthogonality : ', np.dot(x,z))
    
    # choose the parameter for respecting the euclidian norm
    #theta has a uniform distribution, not lambda !
    theta = np.random.uniform(np.arccos(1 - (a*a/2)), np.arccos(1 - (b*b/2)), 1)
    lambd = np.sqrt(-1 + 1/(np.cos(theta)*np.cos(theta)))
    y = x + lambd*z
    return x,  y/np.linalg.norm(y)




def checker(a, b, d, N):
    """
         test function to check whether result is relevant regarding the imposed bounds
    """
    for i in xrange(N):
        x, y = random_points_with_distance_in(a, b, d)
        distance = scipy.spatial.distance.euclidean(x, y)
        
        if (not ((distance >= a) and (distance <= b))):
            print ('error :', distance)
        #assert ((distance >= a) and (distance <= b))

#checker(0.2, 0.4, 32, 1000)

# *** test to show that uniform is faster than normal
#t0 = time.time()
#for i in xrange(10000):
#    a = np.random.normal(0.0, 1.0, 1)
#t1 = time.time()
#for i in xrange(10000):
#    a = np.random.uniform(-1, 1, 1)
#t2 = time.time()
#print('normal :', t1 - t0)
#print('uniform: ', t2 - t1)

     
def do_the_job(n_tests, distance_list, N, k_hash, d, d_pr, kernel, hash_func_list, bonus, OMEGA, OMEGA_CONJ):
    """
    do the job, I mean, compute a list of tuples of probabilities
    """
    
    res_avg = np.zeros(len(distance_list)-1)
    
    for j in xrange(n_tests):

        print('j', j)
        hf = HashFunction(k_hash, d, d_pr, kernel, hash_func_list, bonus)
        
        #WARNING: add np.sqrt(2) at the end of distance_list
        res = np.zeros(len(distance_list)-1)
        
        for i in xrange(len(res)):
            
            a = distance_list[i]
            b = distance_list[i+1]
            
            print('i', i)
            
            for ii in xrange(N):
                
                x,y = random_points_with_distance_in(a, b, d)
                #distance = scipy.spatial.distance.euclidean(x,y)
                #assert ((distance >= a) and (distance <= b))
                h1 = hf.general_hashing(x, OMEGA, OMEGA_CONJ)
                h2 = hf.general_hashing(y, OMEGA, OMEGA_CONJ)
                res[i] += (h1 == h2)
                
            res[i] /= N
        
        res_avg += res
    
    res_avg /= n_tests
    return res_avg    
    
  
  
  
  
if __name__ == '__main__':
    
    k_hash = 1
    kernel = cross_polytope_hashing_basic
    # useless
    c = 2
    R = 1 / math.sqrt(c)
    p1 = 0.9
    p2 = 0.1          
    
    # list of distances considered betwen 2 points: list of intervals
    distance_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, np.sqrt(2)]
    
    # number of points per interval
    N = 20000
    N = 200
    
    d_list = [128, 256]
    d_pr_list = [256, 128, 64, 32]
 
    #TODO:   
    d = d_list[1]
    d_pr = d_pr_list[2]
#    d = 2**11
#    d_pr = 1024

    # global variables, precomputed
    j = complex(0.0, 1.0)
    OMEGA = np.asarray([cmath.exp( j * cmath.pi * i / d) for i in xrange(d)])
    OMEGA_CONJ = np.conjugate(OMEGA)
    
    
    n_tests = 100



    print(' *** random ***')
    bonus = []
    hash_func_list = ['random']
    for i in xrange(len(hash_func_list)):
        bonus.append({})
 
    y1 = do_the_job(n_tests, distance_list, N, k_hash, d, d_pr, kernel, hash_func_list, bonus, OMEGA, OMEGA_CONJ)
    print('y1', y1)
     
   
    print('*** HD3HD2HD1 ***')
    bonus = []
    hash_func_list = ['hadamard', 'hadamard', 'hadamard']
    for i in xrange(len(hash_func_list)):
        bonus.append({})
        bonus[i]['nblocks'] = 1 # because we have 3 blocks at the end anyway  
 
    y2 = do_the_job(n_tests, distance_list, N, k_hash, d, d_pr, kernel, hash_func_list, bonus, OMEGA, OMEGA_CONJ)
    print('y2', y2)   
   
   
    print('*** Skew-Circ D2 HD1 *** ' )
    hash_func_list = ['skew_circulant', 'diagonal', 'hadamard']
    bonus = []
    for i in xrange(len(hash_func_list)):
        bonus.append({})
        if i == len(hash_func_list) - 1:
            bonus[i]['nblocks'] = 1    

    y3 = do_the_job(n_tests, distance_list, N, k_hash, d, d_pr, kernel, hash_func_list, bonus, OMEGA, OMEGA_CONJ)
    print('y3', y3) 
    
    
    print( '*** HDiag(Gaussian) HD2 HD1 ***')
    bonus = []
    hash_func_list = ['pure_hadamard', 'diagonal_gaussian', 'hadamard', 'hadamard']
    for i in xrange(len(hash_func_list)):
        bonus.append({})
        if i > 1:
            bonus[i]['nblocks'] = 1 # because we have 3 blocks at the end anyway     
            
    y4 = do_the_job(n_tests, distance_list, N, k_hash, d, d_pr, kernel, hash_func_list, bonus, OMEGA, OMEGA_CONJ)
    print('y4', y4) 
    
    

    print( '*** T D2 HD1 ***')
    bonus = []
    hash_func_list = ['toeplitz', 'diagonal', 'hadamard']
    for i in xrange(len(hash_func_list)):
        bonus.append({})
        if i > 1 :
            bonus[i]['nblocks'] = 1 # because we have 3 blocks at the end anyway 
            
    y5 = do_the_job(n_tests, distance_list, N, k_hash, d, d_pr, kernel, hash_func_list, bonus, OMEGA, OMEGA_CONJ)
    print('y5', y5) 
    

    print('*** Circ K2 K1 ***')
    bonus = []
    hash_func_list = ['circulant', 'kronecker', 'kronecker']
    for i in xrange(len(hash_func_list)):
        bonus.append({})
        if i != 0:
            bonus[i]['discrete'] = True
            bonus[i]['de'] = 2
    y6 = do_the_job(n_tests, distance_list, N, k_hash, d, d_pr, kernel, hash_func_list, bonus, OMEGA, OMEGA_CONJ)
    print('y6', y6) 
    
    
    # STORE THE RESULTS FOR PLOTTING
    x = distance_list[1:]
    X = np.asarray([x, y1, y6, y5, y3, y4, y2]).T
    print(X)    
    np.savetxt("collision256-64.csv", X, delimiter=",")
    
    print('To the file collision256-64.csv, add as a header: distance,G,GCIRCULANTEK2K1,GTOEPLITZD2HD1,GSKEWD2HD1,HGAUSSHD2HD1,HD3HD2HD1')
