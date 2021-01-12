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

# +
import tensorflow as tf
import numpy as np
import gpflow
from ..inducing_variables.laplacian_dirichlet_features import LaplacianDirichletFeatures

from gpflow.base import TensorLike
# from gpflow import covariances as cov
from gpflow.config import default_float
from .covariances import TimeDomainKuf

Diag = tf.linalg.LinearOperatorDiag
# +
@TimeDomainKuf.register(LaplacianDirichletFeatures, gpflow.kernels.Matern12, TensorLike)
def Kuf_matern12_ldf_time_domain(inducing_variable, kernel, X):
    inds, ω0, d, L = (lambda u: (u.inds, u.ω0, u.d, u.L))(inducing_variable)
    eigen_frequencies = tf.cast(inds+1, gpflow.default_float()) * ω0[None,:]
    arg1 = tf.transpose(X + L[None,:])[None,:,:] #1xDxN
    arg2 = eigen_frequencies[:,:,None] #UxDx1
    Kuf = tf.math.reduce_prod(tf.sin(arg2*arg1)/(tf.sqrt(L)[None,:,None]),-2) #UxN

    # Unlike Variational Fourier Features, the approximation is only
    # valid inside the region. Extrapolation outside is just not what
    # this thing is built for.
    return Kuf

@TimeDomainKuf.register(LaplacianDirichletFeatures, gpflow.kernels.Matern32, TensorLike)
def Kuf_matern32_ldf_time_domain(inducing_variable, kernel, X):
    inds, ω0, d, L = (lambda u: (u.inds, u.ω0, u.d, u.L))(inducing_variable)
    eigen_frequencies = tf.cast(inds+1, gpflow.default_float()) * ω0[None,:]
    arg1 = tf.transpose(X + L[None,:])[None,:,:] #1xDxN
    arg2 = eigen_frequencies[:,:,None] #UxDx1
    Kuf = tf.math.reduce_prod(tf.sin(arg2*arg1)/(tf.sqrt(L)[None,:,None]),-2) #UxN

    # Unlike Variational Fourier Features, the approximation is only
    # valid inside the region. Extrapolation outside is just not what
    # this thing is built for.
    return Kuf

@TimeDomainKuf.register(LaplacianDirichletFeatures, gpflow.kernels.Matern52, TensorLike)
def Kuf_matern52_ldf_time_domain(inducing_variable, kernel, X):
    inds, ω0, d, L = (lambda u: (u.inds, u.ω0, u.d, u.L))(inducing_variable)
    eigen_frequencies = tf.cast(inds+1, gpflow.default_float()) * ω0[None,:]
    arg1 = tf.transpose(X + L[None,:])[None,:,:] #1xDxN
    arg2 = eigen_frequencies[:,:,None] #UxDx1
    Kuf = tf.math.reduce_prod(tf.sin(arg2*arg1)/(tf.sqrt(L)[None,:,None]),-2) #UxN

    # Unlike Variational Fourier Features, the approximation is only
    # valid inside the region. Extrapolation outside is just not what
    # this thing is built for.
    return Kuf

@TimeDomainKuf.register(LaplacianDirichletFeatures, SeparableMatern12, TensorLike)
def Kuf_sep_matern12_ldf_time_domain(inducing_variable, kernel, X):
    inds, ω0, d, L = (lambda u: (u.inds, u.ω0, u.d, u.L))(inducing_variable)
    eigen_frequencies = tf.cast(inds+1, gpflow.default_float()) * ω0[None,:]
    arg1 = tf.transpose(X + L[None,:])[None,:,:] #1xDxN
    arg2 = eigen_frequencies[:,:,None] #UxDx1
    Kuf = tf.math.reduce_prod(tf.sin(arg2*arg1)/(tf.sqrt(L)[None,:,None]),-2) #UxN

    # Unlike Variational Fourier Features, the approximation is only
    # valid inside the region. Extrapolation outside is just not what
    # this thing is built for.
    return Kuf

@TimeDomainKuf.register(LaplacianDirichletFeatures, SeparableMatern32, TensorLike)
def Kuf_sep_matern32_ldf_time_domain(inducing_variable, kernel, X):
    inds, ω0, d, L = (lambda u: (u.inds, u.ω0, u.d, u.L))(inducing_variable)
    eigen_frequencies = tf.cast(inds+1, gpflow.default_float()) * ω0[None,:]
    arg1 = tf.transpose(X + L[None,:])[None,:,:] #1xDxN
    arg2 = eigen_frequencies[:,:,None] #UxDx1
    Kuf = tf.math.reduce_prod(tf.sin(arg2*arg1)/(tf.sqrt(L)[None,:,None]),-2) #UxN

    # Unlike Variational Fourier Features, the approximation is only
    # valid inside the region. Extrapolation outside is just not what
    # this thing is built for.
    return Kuf

@TimeDomainKuf.register(LaplacianDirichletFeatures, SeparableMatern52, TensorLike)
def Kuf_sep_matern52_ldf_time_domain(inducing_variable, kernel, X):
    inds, ω0, d, L = (lambda u: (u.inds, u.ω0, u.d, u.L))(inducing_variable)
    eigen_frequencies = tf.cast(inds+1, gpflow.default_float()) * ω0[None,:]
    arg1 = tf.transpose(X + L[None,:])[None,:,:] #1xDxN
    arg2 = eigen_frequencies[:,:,None] #UxDx1
    Kuf = tf.math.reduce_prod(tf.sin(arg2*arg1)/(tf.sqrt(L)[None,:,None]),-2) #UxN

    # Unlike Variational Fourier Features, the approximation is only
    # valid inside the region. Extrapolation outside is just not what
    # this thing is built for.
    return Kuf

@TimeDomainKuf.register(LaplacianDirichletFeatures, gpflow.kernels.RBF, TensorLike)
def Kuf_RBF_ldf_time_domain(inducing_variable, kernel, X):
    inds, ω0, d, L = (lambda u: (u.inds, u.ω0, u.d, u.L))(inducing_variable)
    eigen_frequencies = tf.cast(inds+1, gpflow.default_float()) * ω0[None,:]
    arg1 = tf.transpose(X + L[None,:])[None,:,:] #1xDxN
    arg2 = eigen_frequencies[:,:,None] #UxDx1
    Kuf = tf.math.reduce_prod(tf.sin(arg2*arg1)/(tf.sqrt(L)[None,:,None]),-2) #UxN

    # Unlike Variational Fourier Features, the approximation is only
    # valid inside the region. Extrapolation outside is just not what
    # this thing is built for.
    return Kuf

