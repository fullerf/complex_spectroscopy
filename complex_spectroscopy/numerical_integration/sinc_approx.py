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
from .util import dtype_to_ctype
from gpflow import default_float

__all__ = ['approx_sinc', 'sinc_scaled_components', 'sinc_unscaled_components']


# +
def sinc_approx_poles(N):
    ctype = dtype_to_ctype(default_float())
    dtype = default_float()
    k = tf.cast(tf.range(0,N),dtype)
    exponent = tf.cast((1+2*k)/N,ctype)
    return 0.5*1j**(exponent)

def sinc_scaled_components(x, N):
    """
    returns a matrix R(x) such that:
    sinc(x) ~= R(x).sum(-1)
    """
    ctype = dtype_to_ctype(default_float())
    dtype = default_float()
    pk = sinc_approx_poles(N) #positive poles only
    num = -2*np.pi*1j*tf.exp(1j*tf.cast(2*np.pi*tf.abs(tf.expand_dims(x,-1)),ctype)*tf.expand_dims(pk,0))
    denom1 = -tf.expand_dims(pk,-1)-tf.expand_dims(pk,0)
    denom2 = tf.linalg.set_diag(-tf.expand_dims(pk,-1)+tf.expand_dims(pk,0),tf.ones(N,ctype))
    denom = tf.reduce_prod(4*denom1*denom2,-1)
    return num/tf.expand_dims(denom,0)

def sinc_unscaled_components(x, N):
    """
    This function returns R and Γ such that:
    
    sinc(x) ~= R(x,q) @ Γ(q), i.e. R are the complex exponential basis functions we expanded sinc with and
                Γ are the N coefficients of those basis functions
    """
    ctype = dtype_to_ctype(default_float())
    dtype = default_float()
    pk = sinc_approx_poles(N) #positive poles only
    num = -2*np.pi*1j*tf.exp(1j*tf.cast(2*np.pi*tf.abs(tf.expand_dims(x,-1)),ctype)*tf.expand_dims(pk,0))
    denom1 = -tf.expand_dims(pk,-1)-tf.expand_dims(pk,0)
    denom2 = tf.linalg.set_diag(-tf.expand_dims(pk,-1)+tf.expand_dims(pk,0),tf.ones(N,ctype))
    denom = tf.reduce_prod(4*denom1*denom2,-1)
    return num, 1/denom

def approx_sinc(x, N):
    """
    Approximates a sinc function using N complex exponentials
    """
    c = sinc_scaled_components(x, N)
    return tf.reduce_sum(c,-1)
# -


