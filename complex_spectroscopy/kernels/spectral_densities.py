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
import tensorflow_probability as tfp
import numpy as np

__all__ = ['MaternSpectralDensityND', 'RBFSpectralDensityND', 'SeparableMaternSpectralDensityND']


# +
def MaternSpectralDensity(ω, ν: float, D: int, l: float):
    """
    See Ch4 Rasmussen & Williams: Gaussian Processes for Machine Learning
    """
#     t1 = (2.**D)*(np.pi**(D/2))*tf.math.exp(tf.math.lgamma(ν + D/2) - tf.math.lgamma(ν))
#     t2 = ((2*ν)**ν)/(l**(2*ν))
#     t3 = ((2*ν)/(l**2)+4*(np.pi**2)*(f**2))**(-(ν+D/2))
    t1 = l**D*(2*np.pi)**(D/2)*(ν**ν)
    t2 = (ν+((l*ω)**2)/2)**(-D/2-ν)
    t3 = tf.math.exp(tf.math.lgamma(ν + D/2) - tf.math.lgamma(ν))
    return t1*t2*t3

def MaternSpectralDensityND(Ω: tf.Tensor, ν: float, D: int, l: tf.Tensor):
    """
    A multi-dimensional generalization of the Matern Spectral Density.
    """
    if len(tf.shape(l)) == 0:
        assert D == 1
        ω = tf.sqrt(tf.math.reduce_sum(tf.math.square(Ω*l),-1))
    else:
        assert tf.shape(l)[-1]==D
        ω = tf.sqrt(tf.math.reduce_sum(tf.math.square(Ω*l[None,:]),-1))
    return MaternSpectralDensity(ω, ν, D, 1.)*tf.math.reduce_prod(l)

def RBFSpectralDensityND(Ω: tf.Tensor, l: tf.Tensor):
    if len(tf.shape(l)) == 0:
        t1 = np.sqrt(2*np.pi)*l
        t2 = tf.math.exp(-tf.square(l*Ω)/2)
        r = t1*tf.math.reduce_prod(t2,-1) 
    else:
        t1 = np.sqrt(2*np.pi)*l
        t2 = tf.math.exp(-tf.square(l[None,:]*Ω)/2)
        r = tf.math.reduce_prod(t1[None,:]*t2,-1)
    return r

def SeparableMaternSpectralDensityND(Ω: tf.Tensor, ν: float, D: int, l: tf.Tensor):
    """
    A multi-dimensional generalization of the Matern Spectral Density.
    """
    assert len(l)==D
    Ωs = Ω*l[None,:]
    ωs = tf.unstack(Ωs,tf.shape(Ω)[-1],axis=-1)
    Ss = MaternSpectralDensity(Ωs, ν, 1, 1.)
    S = tf.math.reduce_prod(Ss,-1)
    return S*tf.math.reduce_prod(l)
