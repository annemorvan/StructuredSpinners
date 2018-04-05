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
#import scipy.spatial.distance
#import scipy.io
#import os
#import matplotlib.pyplot as plt
#import types
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
                self._A = np.random.normal(0.0, 1.0, (self._d_pr,self._d))
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




def test_neural_network_speedups(d, size_1, blocks_list, bonus, n_tests = 1):
    
    # OMP_NUM_THREADS=1 python neural_network.py
    
    print("measure speedups with neural network")
    
    s = reLU 
    
    # global variables, precomputed
    j = complex(0.0, 1.0)
    OMEGA = np.asarray([cmath.exp( j * cmath.pi * i / d) for i in xrange(d)])
    OMEGA_CONJ = np.conjugate(OMEGA)
    
    # initial vector
    x = np.random.normal(0.0, 1.0, d)
    
    b1 = np.random.normal(0.0, 1.0, size_1)
    b2 = np.random.normal(0.0, 1.0, 10)
    
    t_avg = 0
    
    for i in xrange(n_tests):
        
        if i % 10 == 0:
            print('i', i)

        struct1 = StructuredMatrix(d, size_1, s, blocks_list, bonus)
        struct2 = StructuredMatrix(size_1, 10, s, blocks_list, bonus)
            
        t = time.time()
        
        res = struct1.structured_product(x, OMEGA, OMEGA_CONJ) 
        assert res.shape == (size_1, )
        res += b1
        res = s(res)
        res = struct2.structured_product(res, OMEGA, OMEGA_CONJ) 
        assert res.shape == (10,)
        res += b2
            
        t = time.time() - t
        t_avg += t
        
        del struct1
        del struct2
    
    
    return t_avg/n_tests


















  
  
if __name__ == '__main__':


#==============================================================================
# pick up a structured matrix
#==============================================================================



    
  
    



            


#==============================================================================
#   ALGORITHM
#==============================================================================

    d = 1024
    n_tests = 100
    
    size_1_list = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    # for random matrix: 
    # OMP_NUM_THREADS=1 python neural_network_speedup.py

    # *** blocks list ***

    print('*** RANDOM ***')
    blocks_list = ['random']
    bonus = []
    for i in xrange(len(blocks_list)):
        bonus.append({})
    
    res_unstructured = []
    
    for i in xrange(len(size_1_list)): 
        size_1 = size_1_list[i]    
        t1 = test_neural_network_speedups(d, size_1, blocks_list, bonus, n_tests)
        res_unstructured.append(t1)
        print('t1', t1)



    print('*** HD3HD2HD1 ***')
    bonus = []
    blocks_list = ['hadamard', 'hadamard', 'hadamard']
    for i in xrange(len(blocks_list)):
        bonus.append({})
        bonus[i]['nblocks'] = 1 # because we have 3 blocks at the end anyway  
        
    res_structured = []

    for i in xrange(len(size_1_list)): 
        size_1 = size_1_list[i]
        t2 = test_neural_network_speedups(d, size_1, blocks_list, bonus, n_tests)
        res_structured.append(t2)
        print('t2, size_1', t2)    
    
    print('\ndimensions')
    print(size_1_list)
    
    print('\ntime for unstructured matrix')
    print(res_unstructured)

    print('\ntime for structured matrix')
    print(res_structured)
    



