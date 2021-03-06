# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 18:50:14 2016

@author: amorvan
"""
from __future__ import division

import StructuredMatrices

import numpy as np
import math, cmath
import time
import scipy.spatial.distance
import scipy.io
import os
#import matplotlib.pyplot as plt
import types
#import sys 
#import fht
import ffht



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



#def cross_polytope(x):
#    """
#    Find the closest neighbor to x from basis {+-e_i}
#    => find the largest coefficient
#    
#    x is supposed to be normalized
#    """
#    idx = np.argsort(np.abs(x), axis = None)
#    idx = idx[-1]
#    y = np.zeros(x.shape)
#    y[idx] = 1
#    if x[idx] < 0:
#        y[idx] *= -1
#    return y


def target_norm(dim):
    """
    To be used for discrete matrices in order to have each row with a norm
    drawn from the distribution of norm of Gaussian vectors
    """
    tn = np.random.normal(0.0, 1.0, dim)
    tn = np.linalg.norm(tn)
    return tn
    
#def test_target_norm():
#    dim = 128
#    target_norm(dim)
#    
#test_target_norm()
#sys.exit(0)
    
    
#==============================================================================
# kernels
#==============================================================================

def clean_cos(cos_angle):
    """
    to be sure to have a cosinus between -1 and 1
    """
    return min(1,max(cos_angle,-1))
    

def gaussian_kernel(x, y, sigma):
    return math.exp( - (np.linalg.norm(x-y, ord = 2) ** 2) / (2 * (sigma ** 2)) )
    
  
def angular_kernel(x,y, *sigma):
    
    # uncomment things when x,y are not drawn from the hypersphere

    #norm_x = np.linalg.norm(x, ord = 2)
    #norm_y = np.linalg.norm(y, ord = 2)
#    theta = math.acos(clean_cos(np.dot(x,y) / ( norm_x * norm_y)))
    #print('np.dot(x,y)', np.dot(x,y), norm_x, norm_y)
    theta = math.acos(clean_cos(np.dot(x,y)))
    return 1 - theta / math.pi
    
    # x and y are drawn from hypersphere!! Faster?
    #return 1 - math.acos(np.dot(x,y)) / math.pi
 

   
def arccosine_kernel(x,y, *sigma):
    
    # uncomment things when x,y are not drawn from the hypersphere
    
#    norm_x = np.linalg.norm(x, ord = 2)
#    norm_y = np.linalg.norm(y, ord = 2)
#    theta = math.acos(clean_cos(np.dot(x,y) / ( norm_x * norm_y)))
#    return norm_x * norm_y * (math.sin(theta) + (math.pi - theta) * math.cos(theta)) / math.pi

    # x and y are drawn from hypersphere!! Faster?
    #print('dot', np.dot(x,y) )
    theta = math.acos(clean_cos(np.dot(x,y)))
    #print('theta', theta * 180 / math.pi)
    return (math.sin(theta) + (math.pi - theta) * math.cos(theta)) / (2 * math.pi)




#==============================================================================
# non-linearity functions    
#==============================================================================

def complex_exponential(x, sigma): 
    j = complex(0.0, 1.0)
    return np.exp( -(j * x) / sigma)
    
#def sign(x, *sigma):
#    x_bin = np.zeros(x.shape, dtype = 'int')
#    x_bin[x >= 0] = 1
#    return x_bin


def sign11(x, *sigma):
    """
    version which returns -1 or 1
    """
    x_bin = np.ones(x.shape, dtype = 'int')
    x_bin[x < 0] = -1
    return x_bin
 
    
def reLU(x, *sigma):
    #http://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy
    #return x * (x > 0)
    return np.maximum(x,0)
    #return np.log10( np.ones(x.shape) + np.exp(x) )
    





#==============================================================================
# Block class    
#==============================================================================
class Block(object):

    """
    A = B1 B2 B3
    
    Each matrix Bi is represented with corresponding information by a Block class object
    
    Parameters:
    * d : data dimension
    * hash_func : name of the structured matrix for this block
    
    """
    
    def __init__(self, d, d_pr, hash_func, bonus):

        # intialization

        # /!\ the following parameters are one parameter if d_pr <= d
        # or list of parameters if d_pr > d
        # example: for a circulant matrix :
        # div = d_pr// d (euclidean division)
        # so we need div columns parameter to compute the rectangular circulant matrix

        self._d = d  # number of columns
        self._d_pr = d_pr # number of rows
        self._hash_func = hash_func
        
        
        
        self._A = None
        
        self._col = None
        self._row = None    

        # specific to an HDi block
        self._nblocks = None
        self._D = None

        # specific to a low-displacement rank block
        self._r = None
        if self._r:
            assert self._r < self._d
        self._G = None
        self._H = None
        
        # specific to a kronecker block
        self._discrete = None # boolean
        self._de = None
        self._A_list = None
 
        # to be sure I forgot no cases
        self._check = False      
     
        assert self._hash_func in ['random', 'circulant', 'skew_circulant', 'toeplitz', 'hankel', 'pure_hadamard', 'hadamard', 'low_displacement_rank', 'kronecker', 'diagonal', 'diagonal_kronecker', 'diagonal_gaussian' ]
        
        # filling necessary variables
        # TODO: simplify the code by reunification of the two cases

        # less rows than columns
        if self._d_pr <= self._d:
        
            if self._hash_func == 'random':
                self._A = np.random.normal(0.0, 1.0, (self._d_pr,self._d))
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
                self._G = np.random.normal(0.0, 1.0, (self._d,self._r))
                self._H = np.random.normal(0.0, 1.0, (self._d,self._r)) 
                
                #TODO: 
                # each column should be 5-sparse for G50C => 10%
                # UGLY
                for i in xrange(self._r):
                    zeros = np.random.randint(0, self._r, size = int(0.1 *self._d) )
                    for zero in zeros:
                        self._H[zero, i] = 0
                
                
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

            if self._hash_func == 'diagonal_kronecker':
                self._D = StructuredMatrices.compute_kronecker_vector(self._d, n_lev = 3, n_bits = 2, discrete = True)
                self._check = True
            
            if self._hash_func == 'diagonal_gaussian':
                self._D = np.random.normal(0.0, 1.0, self._d)
                self._check = True
                
            if self._hash_func == 'pure_hadamard':
                self._check = True
                # nothing to do
                pass
            
        else:
             # d_pr > d           
            div = self._d_pr // self._d # number of parameters
            
            if self._hash_func == 'random':
                self._A = np.random.normal(0.0, 1.0, (self._d_pr, self._d))
                self._check = True
            
            if self._hash_func == 'circulant' or hash_func == 'skew_circulant' or hash_func == 'toeplitz' or hash_func == 'hankel':
                self._col = [np.random.normal(0.0, 1.0, self._d) for i in xrange(div)]
                self._check = True

            if self._hash_func == 'hadamard':
                assert 'nblocks' in bonus
                self._nblocks = bonus['nblocks']
                self._D = [ StructuredMatrices.compute_diagonal_matrices(self._d, self._nblocks) for i in xrange(div)]
                self._check = True
                # do not compute Hadamard matrix in advance
 
            if self._hash_func == 'toeplitz':
                self._row = [np.random.normal(0.0, 1.0, self._d)   for i in xrange(div)]
                for i in xrange(div):                
                    self._row[i][0] = self._col[i][0]   
                self._check = True

            if self._hash_func == 'hankel':
                self._row = [ np.random.normal(0.0, 1.0, self._d) for i in xrange(div) ]
                for i in xrange(div):
                    self._row[i][0] = self._col[i][self._d - 1] 
                self._check = True
                    
            if self._hash_func == 'low_displacement_rank':
                assert 'r' in bonus
                self._r = bonus['r']
                self._G = [ np.random.normal(0.0, 1.0, (self._d,self._r)) for i in xrange(div)]
                self._H = [ np.random.normal(0.0, 1.0, (self._d,self._r)) for i in xrange(div)]  

                #TODO: 
                # each column should be 5-sparse for G50C => 10%
                # UGLY
                for j in xrange(div):
                    for i in xrange(self._r):
                        zeros = np.random.randint(0, self._r, size = int(0.1 *self._d) )
                        for zero in zeros:
                            self._H[j][zero, i] = 0

                self._check = True                 

            if self._hash_func == 'kronecker':
                assert 'discrete' in bonus 
                assert 'de' in bonus
                self._discrete = bonus['discrete']
                self._de = bonus['de'] 
                self._A_list = [ StructuredMatrices.kronecker( int(math.log(self._d, 2)), self._de, self._discrete) for i in xrange(div)]
                self._check = True
                
            if self._hash_func == 'diagonal':
                self._D = [ StructuredMatrices.compute_diagonal_matrices(self._d, 1).reshape(self._d,) for i in xrange(div)]
                self._check = True
            
            if self._hash_func == 'diagonal_kronecker':
                self._D = [ StructuredMatrices.compute_kronecker_vector(self._d, n_lev = 3, n_bits = 2, discrete = True) for i in xrange(div)]
                self._check = True

            if self._hash_func == 'diagonal_gaussian':
                self._D = [ np.random.normal(0.0, 1.0, self._d) for i in xrange(div)]
                self._check = True

            if self._hash_func == 'pure_hadamard':
                self._check = True
                # nothing to do
                pass
        
        # to be sure we entered at least in one if
        assert self._check 






class StructuredMatrix(object):
    
    def __init__(self, d, d_pr, kernel, blocks_list, bonus):  
        
        self._d = d  
        self._d_pr = d_pr
        self._kernel = kernel 
        self._blocks_list = blocks_list
        
        self._normalization = False
        if self._kernel == arccosine_kernel and self._blocks_list[0] == 'hadamard':
            print('NORMALIZATION IS NECESSARY')
            self._normalization = True
        
        #TODO: just a test
        #self._normalization = True
        
        self._norm_factor_list = None
        if self._normalization:
            #dim = self._d_pr
            self._norm_factor_list = np.asarray([ target_norm(self._d) for i in xrange(self._d_pr)])
            #self._norm_factor_list = np.ones(dim)
                
        # information corresponding to 3 different blocks
        self._BBB = [Block(self._d , self._d_pr, self._blocks_list[i], bonus[i]) for i in xrange(len(blocks_list))]



    def structured_product(self, x, OMEGA, OMEGA_CONJ):
        
        y = np.copy(x)
        #print(self._d_pr, self._d)
        if self._d_pr <= self._d:
            
            for block in self._BBB[::-1]:
        
                # matrix-vector product
                if block._hash_func == 'random':
                    y = np.dot(block._A, y) 
                    
                if block._hash_func == 'circulant': 
                    y = StructuredMatrices.circulant_product(block._col, y)
                
                if block._hash_func == 'skew_circulant':
                    y = StructuredMatrices.skew_circulant_product(block._col , y, OMEGA, OMEGA_CONJ)
                
                if block._hash_func == 'toeplitz':
                    y = StructuredMatrices.toeplitz_product(block._col, block._row, y)
                   
                if block._hash_func == 'hadamard':
                    y = StructuredMatrices.hadamard_cross_polytope_product(y, block._D, block._nblocks)
                    
                if block._hash_func == 'hankel':
                    y = StructuredMatrices.hankel_product(block._col, block._row, y)
                
                if block._hash_func == 'low_displacement_rank':        
                    y = StructuredMatrices.low_displacement_product(block._G, block._H, block._r, y, OMEGA, OMEGA_CONJ)
                    
                if block._hash_func == 'kronecker': 
                    y = StructuredMatrices.kronecker_product_rec(block._A_list, y)
                                    
                if block._hash_func == 'pure_hadamard':
                    #y = fht.fht1(y)
                    a = ffht.create_aligned(y.shape[0], np.float32)
                    np.copyto(a, y)
                    ffht.fht(a, min(y.shape[0],  1024, 2048))   
                    y = a
                if block._hash_func == 'diagonal' or block._hash_func == 'diagonal_kronecker' or block._hash_func == 'diagonal_gaussian':
                    y = block._D * y
                    
                #print(block, y.shape)
        
            # all blocks have been handled.
                
            # dimensionality reduction
            y = y[:self._d_pr]  
            if self._normalization:
                y *= self._norm_factor_list
            return y
        
        else:
            
            if len(self._BBB) == 1 and self._BBB[0]._hash_func == 'random':
                y = np.dot(self._BBB[0]._A, y)
                return y
            
            y_global = np.zeros(self._d_pr)
            
            div = self._d_pr // self._d # number of parameters
            div_real = self._d_pr / self._d    
            #print(div, div_real, self._d_pr, self._d)
            assert div == div_real # we handle only this case
            # /!\ + d_pr should be an integral power of 2!!!
            for i_div in xrange(div):            

                #print('i_div', i_div)
                y = np.copy(x)
                
                # if div_real != div, the last mini block should be truncated => never happened because of the case of hadamard
                # split  y into small vectors  and reunit all results after
                for block in self._BBB[::-1]:
                    #if i_div == div or i_div < div -1:
                        # easy

                    # matrix-vector product
                        
                    if block._hash_func == 'circulant': 
                        y = StructuredMatrices.circulant_product(block._col[i_div], y)
                    
                    if block._hash_func == 'skew_circulant':
                        y = StructuredMatrices.skew_circulant_product(block._col[i_div] , y, OMEGA, OMEGA_CONJ)
                    
                    if block._hash_func == 'toeplitz':
                        y = StructuredMatrices.toeplitz_product(block._col[i_div], block._row[i_div], y)
                       
                    if block._hash_func == 'hadamard':
                        y = StructuredMatrices.hadamard_cross_polytope_product(y, block._D[i_div], block._nblocks)
                        
                    if block._hash_func == 'hankel':
                        y = StructuredMatrices.hankel_product(block._col[i_div], block._row[i_div], y)
                    
                    if block._hash_func == 'low_displacement_rank':        
                        y = StructuredMatrices.low_displacement_product(block._G[i_div], block._H[i_div], block._r, y, OMEGA, OMEGA_CONJ)
                        
                    if block._hash_func == 'kronecker': 
                        y = StructuredMatrices.kronecker_product_rec(block._A_list[i_div], y)
                                        
                    if block._hash_func == 'pure_hadamard':
                        a = ffht.create_aligned(y.shape[0], np.float32)
                        np.copyto(a, y)
                        ffht.fht(a, min(x.shape[0], 1024, 2048))   
                        y = a
            
                    if block._hash_func == 'diagonal' or block._hash_func == 'diagonal_kronecker' or block._hash_func == 'diagonal_gaussian':
                        y = block._D[i_div] * y 
                
                y_global[i_div * self._d: i_div * self._d + self._d] = y
                #print(y_global)
                if self._normalization:
                    y_global *= self._norm_factor_list
                
            return y_global



#    #print(diff/n_tests)
#    #idx = np.asarray(abscisses).argsort()
#    #print(abscisses)
#    #plt.plot(np.asarray(abscisses)[idx], np.asarray(ordonnees)[idx], 'o-')
#    plt.plot(np.arange(n_tests), diff_list, 'o')    
#    #plt.plot(np.arange(n_tests), ratio_list, '-')
#    #plt.title('Error on kernel approximation: kernel = sign; structured matrix = HD3 HD2 HD1')
#    #plt.title('Error on kernel approximation: kernel = ReLU; structured matrix = S-Circ D2 HD1')
#    plt.title('Error on kernel approximation: kernel = cross-polytope; structured matrix = random')
#    #plt.ylim(-2, 1)
#    plt.xlabel('Number of test', fontsize = 15)
#    plt.ylabel('Error', fontsize = 15)
#    plt.grid()
#    plt.show()
  


def Gram_test(n_tests, X, d, d_pr_list, n, sigma, kernel, s, blocks_list, bonus):
    
    print("Gram matrix reconstruction error")
    # global variables, precomputed
    j = complex(0.0, 1.0)
    OMEGA = np.asarray([cmath.exp( j * cmath.pi * i / d) for i in xrange(d)])
    OMEGA_CONJ = np.conjugate(OMEGA)
        
    load = False
    #sys.exit(0) 
    
    error_list = []

    # 1 - Exact kernel computation
            
    if not load or not os.path.isfile('K.data'):
                        
        # apply kernel, gold results
        K = np.empty((n,n))
        
        for j in xrange(n):
            #print('point j', j)
            #if j % 200 == 0:
            #   print('j', j)
                
            for k in xrange(n):
                
                if j <= k:
                    
                    res = kernel(X[:, j], X[:,k], sigma)
                    K[j,k] = res
                    K[k,j] = res
                    
        #K /= d_pr
                    
        print('K done, mean', K.mean(), 'stdev', K.std())
        #print(K)
        pickle(K, 'K.data')
                
    else:
                
        K = unpickle('K.data')
        #K /= d_pr
        print('K loaded')
      
    for l in xrange(len(d_pr_list)):
        
        d_pr = d_pr_list[l]
        error = 0
        
        print('d_pr', d_pr)
        
        for i in xrange(n_tests):
            # 2 - Approximate kernel computation  
     
            if not load or not os.path.isfile('K_tilde.data') :      
                      
                struct = StructuredMatrix(d, d_pr, s, blocks_list, bonus)
                
                if kernel == gaussian_kernel:  
                    K_tilde = np.empty((n,n), dtype = types.ComplexType)
                else:
                    K_tilde = np.empty((n,n))
		# Un-normalize the Hadamard applied last.
		fact = 1.0
		if blocks_list[0] == 'hadamard' or blocks_list[0] == 'pure_hadamard':
       		   fact = d ** (0.5)

		# Compute random features once for each input vector
		random_features = np.empty((n,d_pr), dtype = types.ComplexType)
		for j in xrange(n):
		   random_features[j] = s(fact * struct.structured_product(X[:,j], OMEGA, OMEGA_CONJ), sigma)

                for j in xrange(n):
                    
#                    if j % 100 == 0:
#                        print('j', j)
                        
                    for k in xrange(n):
                        
                        if j <= k:
                            
                            res1 = random_features[j]
                            res2 = random_features[k]
                             
                            if kernel == angular_kernel:
                                res = 1 - scipy.spatial.distance.hamming(res1,res2)
                            else:
                                res = np.dot(np.conjugate(res1), res2) / d_pr
                                # np.vdot for complex numbers
                            
                            K_tilde[j,k] = res
                            K_tilde[k,j] = res
                            
                #K_tilde /= d_pr
                            
                print('K_tilde done')
                #print(K_tilde)
                pickle(K_tilde, 'K_tilde.data')
                
            else:
                
                K_tilde = unpickle('K_tilde.data')
                #K_tilde /= d_pr
                print('K_tilde loaded')
    
            Diff_K = np.linalg.norm(K - K_tilde, ord = 'fro')
            Diff_K /= np.linalg.norm(K, ord = 'fro')
            error += Diff_K
            print(l, i, 'current error', error / (i + 1), Diff_K)
        
        error_avg = error / n_tests        
        print(l, 'error', error_avg )
        error_list.append(error_avg)
    
    print('error_list', error_list)
    name = ''
    for elt in blocks_list:
        name += elt
    name += '_' + str(kernel)
    name += '.data'
    pickle(error_list, name)
    return error_list


def test_angular(X, d, d_pr, n, sigma, blocks_list, bonus):
    
    print("test_angular")
    
    # *** kernel and non linearity functions ***
    kernel_list = [gaussian_kernel, angular_kernel, arccosine_kernel]
    non_linearity_list = [complex_exponential, sign11, reLU]

    kernel = kernel_list[1]
    s = non_linearity_list[1]
    
    assert kernel_list.index(kernel) == non_linearity_list.index(s)    
    
    # global variables, precomputed
    j = complex(0.0, 1.0)
    OMEGA = np.asarray([cmath.exp( j * cmath.pi * i / d) for i in xrange(d)])
    OMEGA_CONJ = np.conjugate(OMEGA)
        
    
    x1 = X[:, np.random.randint(0, n-1)]
    x2 = X[:, np.random.randint(0, n-1)]
#    x1 = np.zeros(d)
#    x1[2] = 1
#    x2 = np.zeros(d)
#    x2[6] = 1
    
    res_true = kernel(x1, x2, sigma)
    
    struct = StructuredMatrix(d, d_pr, s, blocks_list, bonus)
    res1 = s(struct.structured_product(x1, OMEGA, OMEGA_CONJ), sigma) / ( math.sqrt(d_pr))
    res2 = s(struct.structured_product(x2, OMEGA, OMEGA_CONJ), sigma) / ( math.sqrt(d_pr))
                            
    if kernel == angular_kernel:
        res = 1 - scipy.spatial.distance.hamming(res1,res2)
    else:
        res = np.dot(np.conjugate(res1), res2)
 
    print('res_true', res_true)
    print('res_approx', res)
        



def test_gaussian(X, d, d_pr, n, sigma, blocks_list, bonus):
    
    print("test_gaussian")
    
    # *** kernel and non linearity functions ***
    kernel_list = [gaussian_kernel, angular_kernel, arccosine_kernel]
    non_linearity_list = [complex_exponential, sign11, reLU]

    kernel = kernel_list[0]
    s = non_linearity_list[0]
    
    assert kernel_list.index(kernel) == non_linearity_list.index(s)   
    
    # global variables, precomputed
    j = complex(0.0, 1.0)
    OMEGA = np.asarray([cmath.exp( j * cmath.pi * i / d) for i in xrange(d)])
    OMEGA_CONJ = np.conjugate(OMEGA)
        
    
    x1 = X[:, np.random.randint(0, n-1)]
    x2 = X[:, np.random.randint(0, n-1)]
#    x1 = np.zeros(d)
#    x1[2] = 1
#    x2 = np.zeros(d)
#    x2[6] = 1
    
    res_true = kernel(x1, x2, sigma)
    
    struct = StructuredMatrix(d, d_pr, s, blocks_list, bonus)
    # Un-normalize the Hadamard applied last.
    fact = 1.0
    if blocks_list[0] == 'hadamard' or blocks_list[0] == 'pure_hadamard':
       fact = d ** (0.5)
    res1 = s(fact * struct.structured_product(x1, OMEGA, OMEGA_CONJ), sigma) / ( math.sqrt(d_pr))
    res2 = s(fact * struct.structured_product(x2, OMEGA, OMEGA_CONJ), sigma) / ( math.sqrt(d_pr))
                            
    if kernel == angular_kernel:
        res = 1 - scipy.spatial.distance.hamming(res1,res2)
    else:
        res = np.dot(np.conjugate(res1), res2)
 
    print('res_true', res_true)
    print('res_approx', res)
        
        


def test_arccos(X, d, d_pr, n, sigma, blocks_list, bonus):
    
    print("test_arccos")
    
    # *** kernel and non linearity functions ***
    kernel_list = [gaussian_kernel, angular_kernel, arccosine_kernel]
    non_linearity_list = [complex_exponential, sign11, reLU]

    kernel = kernel_list[2]
    s = non_linearity_list[2]
    
    assert kernel_list.index(kernel) == non_linearity_list.index(s) 
       
    # global variables, precomputed
    j = complex(0.0, 1.0)
    OMEGA = np.asarray([cmath.exp( j * cmath.pi * i / d) for i in xrange(d)])
    OMEGA_CONJ = np.conjugate(OMEGA)
        
    
    x1 = X[:, np.random.randint(0, n-1)]
    x2 = X[:, np.random.randint(0, n-1)]
    #print(np.linalg.norm(x1), np.linalg.norm(x2))
    
#    # (1,0) and (\sqrt(2) / 2, \sqrt(2) / 2), theta = pi /2, kernel = 0.75
#    x1 = np.zeros(d)
#    x1[0] = 1
#    x2 = np.zeros(d)
#    x2[0] = math.sqrt(2) / 2
#    x2[1] = math.sqrt(2) / 2    
#    #x1[2] = 1
    
#    x2 = np.zeros(d)
#    #x2[6] = 1
#    x2[0] = math.sqrt(2) / 2
#    x2[1] = math.sqrt(2) / 2
    
#    # ( sqrt(3) / 2, -1/2) and (sqrt(3) / 2, 1/2) theta = pi/3, kernel = 0.60
#    x1 = np.asarray([math.sqrt(3)/2, -0.5])
#    x2 = np.asarray([math.sqrt(3)/2, 0.5])
    
    res_true = kernel(x1, x2, sigma) 

    print('d_pr', d_pr)
    
    struct = StructuredMatrix(d, d_pr, s, blocks_list, bonus)
    y1 = struct.structured_product(x1, OMEGA, OMEGA_CONJ)
    #print('y1', y1)
    res1 = s(y1, sigma) 
    y2 = struct.structured_product(x2, OMEGA, OMEGA_CONJ)
    #print('y2', y2)
    res2 = s(y2, sigma) 
                           
    if kernel == angular_kernel:
        res = 1 - scipy.spatial.distance.hamming(res1,res2)
    else:
        res = np.dot(res1, res2) / d_pr
 
    print('res_true', res_true)
    print('res_approx', res)
        


def test_gaussian_speedups(X, d, d_pr, n, sigma, blocks_list, bonus):
    
    print("measure speedups with the Gaussian kernel")
    
    # *** parameters ***
    n_tests = 100
    
    # *** kernel and non linearity functions ***
    kernel_list = [gaussian_kernel, angular_kernel, arccosine_kernel]
    non_linearity_list = [complex_exponential, sign11, reLU]

    kernel = kernel_list[0]
    s = non_linearity_list[0]
    
    assert kernel_list.index(kernel) == non_linearity_list.index(s)   
    
    # global variables, precomputed
    j = complex(0.0, 1.0)
    OMEGA = np.asarray([cmath.exp( j * cmath.pi * i / d) for i in xrange(d)])
    OMEGA_CONJ = np.conjugate(OMEGA)
        
    
    x1 = X[:, np.random.randint(0, n-1)]
    x2 = X[:, np.random.randint(0, n-1)]
#    x1 = np.zeros(d)
#    x1[2] = 1
#    x2 = np.zeros(d)
#    x2[6] = 1
    
    t_avg = 0
    for i in xrange(n_tests):
        if i % 10 == 0:
            print('i', i)

        struct = StructuredMatrix(d, d_pr, s, blocks_list, bonus)
            
        t = time.time()
        
        res1 = s(struct.structured_product(x1, OMEGA, OMEGA_CONJ), sigma) / ( math.sqrt(d_pr))
        res2 = s(struct.structured_product(x2, OMEGA, OMEGA_CONJ), sigma) / ( math.sqrt(d_pr))
        
        del struct
                     
        if kernel == angular_kernel:
            res = 1 - scipy.spatial.distance.hamming(res1,res2)
        else:
            res = np.vdot(res1, res2)
            
        t = time.time() - t
        
        t_avg += t
    
    print('t_avg', t_avg/n_tests)

  
  
if __name__ == '__main__':

#==============================================================================
# CHOOSE IF YOU WANT TO TEST GAUSSIAN OR ANGULAR KERNEL
#==============================================================================

    # *** kernel and non linearity functions ***
    kernel_list = [gaussian_kernel, angular_kernel, arccosine_kernel]
    non_linearity_list = [complex_exponential, sign11, reLU]
    
    kernel = kernel_list[1]
    print('Kernel', kernel)
    s = non_linearity_list[1]
    print('Non-linearity', s)    
    assert kernel_list.index(kernel) == non_linearity_list.index(s)
    
#==============================================================================
# CHOOSE YOU DATASET BTW G5OC and USPST 
#==============================================================================

    datasets = ['G50C.mat', 'COIL20.mat', 'USPST.mat']
    dataset = datasets[2] 
    assert dataset != datasets[1]
   
    mat = scipy.io.loadmat(dataset)
    print('DATASET LOADED', dataset)
    print(mat.keys())

    sigma = mat['SIGMA'][0][0]
    print('sigma', sigma)
    X = mat['X']
    n, d = X.shape
    
    pad = True
       

    # /!\ don't forget to pad to have an integral power of 2 for dimension!
    if dataset == 'G50C.mat' and pad:
        D = 2**6 # 64
        Xpad = np.zeros([n,D])
        Xpad[:, :d] = X
        X = Xpad.T
        d = D
    else:
        X = X.T
        
    #TODO:
    # Skipping normalization is needed for RBF, but it breaks angular kernel!
    if kernel == gaussian_kernel:
        normalization_done = True
    elif kernel == angular_kernel:
        normalization_done = False
    else:
        print('Kernel not supported yet!!!')
        
    if not normalization_done :
        assert kernel == angular_kernel
        X /= np.linalg.norm(X, axis = 0).reshape(-1,1).T
        normalization_done = True
        print('NORMALIZED')
    else:
        assert kernel == gaussian_kernel
        print('NOT NORMALIZED')

    
    d, n = X.shape
    print('X.shape', d, n)

    
    # for G5OC from 2**6 to 2**11
    if dataset == 'G50C.mat':
        d_pr_list = [64, 128, 256, 512, 1024, 2048]
        x = [6, 7, 8, 9, 10, 11]
    
    elif dataset == 'USPST.mat':
        # for USPST from 2**8 to 2**13
        d_pr_list = [256, 512, 1024, 2048, 4096, 8192]
        x = [8, 9, 10, 11, 12, 13]
    else:
        d_pr_list = []
        x = []
    
    print('d_pr_list', d_pr_list)



#==============================================================================
# KERNEL TESTS    
#==============================================================================

    print('')
    print('***** KERNEL TESTS *****')

     
    # IT'S WORKING
    print('')
    print('angular kernel')
    if not normalization_done:
        print('ERROR, data is not normalized, it should be for angular kernel')
#TODO:
#    test_angular(X, d, d_pr, n, sigma, blocks_list, bonus)
    
    # IT'S WORKING
    print('')
    print('gaussian kernel')
    assert normalization_done 
    print('No normalization should be done for gaussian kernel')
    #TODO:
#    test_gaussian(X, d, d_pr, n, sigma, blocks_list, bonus)
 

#==============================================================================
# GRAM TESTS
#==============================================================================

    n_tests = 10
    
    print('')
    print('***** GRAMS TESTS *****')
    


    if kernel == gaussian_kernel:
        print('')
        print('--------- gaussian kernel ---------')
      
       
       
        print('*** RANDOM ***')
        blocks_list = ['random']
        bonus = []
        for i in xrange(len(blocks_list)):
            bonus.append({})       
           
        gauss0 = Gram_test(n_tests, X, d, d_pr_list, n, sigma, kernel, s, blocks_list, bonus)
        
        
        
        print('*** HD3HD2HD1 ***')
        bonus = []
        blocks_list = ['hadamard', 'hadamard', 'hadamard']
        for i in xrange(len(blocks_list)):
            bonus.append({})
            bonus[i]['nblocks'] = 1 # because we have 3 blocks at the end anyway           

        gauss1 = Gram_test(n_tests, X, d, d_pr_list, n, sigma, kernel, s, blocks_list, bonus) 
        
            
        print('*** Skew-Circ D2 HD1 *** ' )
        blocks_list = ['skew_circulant', 'diagonal', 'hadamard']
        bonus = []
        for i in xrange(len(blocks_list)):
            bonus.append({})
            if i == len(blocks_list) - 1:
                bonus[i]['nblocks'] = 1        
        
        gauss2 = Gram_test(n_tests, X, d, d_pr_list, n, sigma, kernel, s, blocks_list, bonus) 
        
        
        # *** HDiag(Gaussian) HD2 HD1 ***
        bonus = []
        blocks_list = ['pure_hadamard', 'diagonal_gaussian', 'hadamard', 'hadamard']
        for i in xrange(len(blocks_list)):
            bonus.append({})
            if i > 1:
                bonus[i]['nblocks'] = 1 # because we have 3 blocks at the end anyway 
        gauss3 = Gram_test(n_tests, X, d, d_pr_list, n, sigma, kernel, s, blocks_list, bonus) 
        
        
        
        print('*** Toeplitz D2 HD1 *** ' )
        blocks_list = ['toeplitz', 'diagonal', 'hadamard']
        bonus = []
        for i in xrange(len(blocks_list)):
            bonus.append({})
            if i == len(blocks_list) - 1:
                bonus[i]['nblocks'] = 1
        gauss4 = Gram_test(n_tests, X, d, d_pr_list, n, sigma, kernel, s, blocks_list, bonus) 

 
    
        print(' *** Circ K2 K1 *** ')
        bonus = []
        blocks_list = ['circulant', 'kronecker', 'kronecker']
        for i in xrange(len(blocks_list)):
            bonus.append({})
            if i != 0:
                bonus[i]['discrete'] = True
                bonus[i]['de'] = 2
        gauss5 = Gram_test(n_tests, X, d, d_pr_list, n, sigma, kernel, s, blocks_list, bonus) 

    
        X = np.asarray([  x, gauss0, gauss5, gauss4, gauss2, gauss3, gauss1]).T
        print(X)    
        np.savetxt("gaussiankernel_USPST.csv", X, delimiter=",")
        print('Add manual as a header: n,G,CircK2K1,GTOEPLITZD2HD1,GSKEWD2HD1,HGAUSSHD2HD1,HD3HD2HD1')         
        
    
    elif kernel == angular_kernel:
 
        print('')
        print('--------- angular kernel ---------')
        if not normalization_done:
            print('ERROR, data is not normalized, it should be for angular kernel')


        print('*** RANDOM ***')
        blocks_list = ['random']
        bonus = []
        for i in xrange(len(blocks_list)):
            bonus.append({})       
           
        ang0 = Gram_test(n_tests, X, d, d_pr_list, n, sigma, kernel, s, blocks_list, bonus)
        
        
        
        print('*** HD3HD2HD1 ***')
        bonus = []
        blocks_list = ['hadamard', 'hadamard', 'hadamard']
        for i in xrange(len(blocks_list)):
            bonus.append({})
            bonus[i]['nblocks'] = 1 # because we have 3 blocks at the end anyway           

        ang1 = Gram_test(n_tests, X, d, d_pr_list, n, sigma, kernel, s, blocks_list, bonus) 
        
            
        print('*** Skew-Circ D2 HD1 *** ' )
        blocks_list = ['skew_circulant', 'diagonal', 'hadamard']
        bonus = []
        for i in xrange(len(blocks_list)):
            bonus.append({})
            if i == len(blocks_list) - 1:
                bonus[i]['nblocks'] = 1        
        
        ang2 = Gram_test(n_tests, X, d, d_pr_list, n, sigma, kernel, s, blocks_list, bonus) 
        
        
        print('*** HDiag(Gaussian) HD2 HD1 ***')
        bonus = []
        blocks_list = ['pure_hadamard', 'diagonal_gaussian', 'hadamard', 'hadamard']
        for i in xrange(len(blocks_list)):
            bonus.append({})
            if i > 1:
                bonus[i]['nblocks'] = 1 # because we have 3 blocks at the end anyway 
        ang3 = Gram_test(n_tests, X, d, d_pr_list, n, sigma, kernel, s, blocks_list, bonus) 
        
        
        
        print('*** Toeplitz D2 HD1 *** ' )
        blocks_list = ['toeplitz', 'diagonal', 'hadamard']
        bonus = []
        for i in xrange(len(blocks_list)):
            bonus.append({})
            if i == len(blocks_list) - 1:
                bonus[i]['nblocks'] = 1
        ang4 = Gram_test(n_tests, X, d, d_pr_list, n, sigma, kernel, s, blocks_list, bonus) 

 
    
        print(' *** Circ K2 K1 *** ')
        bonus = []
        blocks_list = ['circulant', 'kronecker', 'kronecker']
        for i in xrange(len(blocks_list)):
            bonus.append({})
            if i != 0:
                bonus[i]['discrete'] = True
                bonus[i]['de'] = 2
        ang5 = Gram_test(n_tests, X, d, d_pr_list, n, sigma, kernel, s, blocks_list, bonus) 

    
        X = np.asarray([  x, ang0, ang5, ang4, ang2, ang3, ang1]).T
        print(X)    
        np.savetxt("angularkernel_USPST.csv", X, delimiter=",")    
        print('Add manual as a header: n,G,CircK2K1,GTOEPLITZD2HD1,GSKEWD2HD1,HGAUSSHD2HD1,HD3HD2HD1')



