# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import tensorflow as tf
import numpy as np
import gpflow
from gpflow.inducing_variables import InducingVariables

__all__ = ['LaplacianDirichletFeatures']


# +
def sortk(x,k, direction='DESCENDING'):
    """for 1D x, returns vals, inds for the top k* vals. 
    
        *returns min(len(x), k) values
    
    """
    k = min(int(k), len(x))
    i = tf.argsort(x,direction=direction)
    return tf.gather(x, i[:k]), i[:k]

def topksum(E, k):
    """
    For a set of eigenvalues supplied by E (each row of E are eigenvalues of a different dim),
    we compute the top k combinations of E (given by the sum of values over dims). Returns
    a set of indices for each dimension as well as the combined eigenvalues
    """
    k = int(k)
    assert len(tf.shape(E)) == 2
    q = tf.shape(E)[1]
    lam, idxs = sortk(E[0,:], k)
    idxs = idxs[None,:]
    for l in E[1:]:
        z = tf.repeat(lam,q) + tf.tile(l,[len(lam)])
        lam, ords = sortk(z, k)
        a = tf.map_fn(lambda x: tf.gather(x, ords//q), idxs)
        b = (ords % q)[None,:]
        idxs = tf.concat([a, b],0)
    return lam, idxs

def bottomksum(E, k):
    """
    For a set of eigenvalues supplied by E (each row of E are eigenvalues of a different dim),
    we compute the smallest k combinations of E (given by the sum of values over dims). Returns
    a set of indices for each dimension as well as the combined eigenvalues
    """
    k = int(k)
    assert len(tf.shape(E)) == 2
    q = tf.shape(E)[1]
    lam, idxs = sortk(E[0,:], k, direction='ASCENDING')
    idxs = idxs[None,:]
    for l in E[1:]:
        z = tf.repeat(lam,q) + tf.tile(l,[len(lam)])
        lam, ords = sortk(z, k, direction='ASCENDING')
        a = tf.map_fn(lambda x: tf.gather(x, ords//q), idxs)
        b = (ords % q)[None,:]
        idxs = tf.concat([a, b],0)
    return lam, idxs

def all_eigs(E):
    q = tf.shape(E)[1]
    lam = E[0,:]
    for l in E[1:]:
        lam = tf.tile(lam,[q]) + tf.repeat(l,len(lam))
    return lam


# -

class LaplacianDirichletFeatures(InducingVariables):
    """
    Implements methods to compute the eigenfunctions/values of the laplacian on a d-dimensional hypercube
    with dirichlet boundary conditions. These functions are essentially sinusoids of a frequency
    determined by the dimension of the cube. Given the dimensions of the cube, we compute the 
    eigen-frequencies so that the top m lowest frequency terms are retained. For a spectral density that
    is strictly decreasing in frequency, as is the case for most standard stationary kernels, this
    selection of frequencies will capture the most variance in the fewest basis functions possible.
    
    Parameters:
    
    d: the number of dimensions of the hypercube
    m: the number of eigenvalues to retain (number of inducing functions)
    L: the hyper cube is defined on [-L1, -L2, ..., -Ld] to [L1, L2, ..., Ld],
     i.e. symmetric bounds with an adjustable size per dimension. 
     Default value for L1-Ld is None,which translates to [-1,-1,...,-1] to
     [1,1,...,1].
    R: the number of extra frequencies we want to compute for the "remainder". Can be
    used for approximations of the content the low-rank approximation missed.
    freq_strategy: one of 'bottom_k_squares' or 'bottom_k', these refer to strategies
     for selecting which frequencies to include.
    start_ind: defaults to 0, but for positive values we skip the first ind frequencies.
     This can be useful for computing residual inducing kernels.
     
    
    """
    def __init__(self, d: int, m: int, R = None, L = None, 
                 freq_max = None, freq_strategy='bottom_k_squares',
                 start_ind=0):
        assert int(d) > 0
        self._d = int(d)
        assert int(m) > 0
        self._m = int(m)
        N = int(m)
        # if int(m) <= 1000:
        #     N = int(m)
        # else:
        #     N = int(max(np.sqrt(m),1000))
        self.N = N
        if L is None:
            if freq_max is not None:
                L = tf.constant(self.d*[(1/float(freq_max))*float(m)*np.pi/2], dtype=gpflow.default_float())
            else:
                L = tf.constant(self.d*[1.], dtype=gpflow.default_float())
        else:
            L = tf.reshape(tf.constant(L, dtype=gpflow.default_float()),(-1))
            assert len(L) == self.d
        start_ind = int(start_ind)
        assert start_ind >= 0
        self._L = tf.stack([tf.constant(float(l), dtype=gpflow.default_float()) for l in L])
        self._piover2 = tf.constant(np.pi/2, dtype=gpflow.default_float())
        self._base_freq = self._piover2/self.L
        if freq_strategy == 'bottom_k_squares':
            _, inds = bottomksum(tf.math.square(self._freqs_per_dim()), m+start_ind)
        elif freq_strategy == 'bottom_k':
            _, inds = bottomksum(self._freqs_per_dim(), m+start_ind)
        else:
            ValueError('unknown frequency selection strategy')
        self._inds = tf.transpose(inds[:,start_ind:])
        if R is not None:
            self._R = int(R)
            self.remainder = LaplacianDirichletFeatures(self.d, int(R), 
                                                    None, 
                                                    L=self.L,
                                                    freq_strategy=freq_strategy,
                                                    start_ind=self.num_inducing
                                                   )
        
    def __len__(self):
        return self._m
    
    @property
    def num_inducing(self):
        return self._m
        
    @property
    def inds(self):
        return self._inds
    
    @property
    def R(self):
        return self._R
    
    @property
    def L(self):
        return self._L
    
    @property
    def d(self):
        return self._d
    
    @property
    def ω0(self):
        return self._base_freq
    
    @property
    def ωs(self):
        return tf.cast(self.inds+1, gpflow.default_float()) * self.ω0[None,:]
    
    @property
    def Ω(self):
        return tf.math.sqrt(tf.math.reduce_sum(tf.square(self.ωs),-1))
        
    def _freqs_per_dim(self):
        """ Returns N frequencies from each dimension """
        ns = tf.tile(tf.cast(tf.range(1,self.N+1), gpflow.default_float())[None,:],[self.d,1])
        return ns*self.ω0[:,None]



