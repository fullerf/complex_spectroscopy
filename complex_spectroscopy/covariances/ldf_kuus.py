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
from ..inducing_variables import LaplacianDirichletFeatures
from ..kernels import *
from gpflow import covariances as cov
Diag = tf.linalg.LinearOperatorDiag

__all__ = []
# +
@cov.Kuu.register(LaplacianDirichletFeatures, gpflow.kernels.Matern12)
def Kuu_matern12_ldf(inducing_variable, kernel, jitter=None):
    """
    Kuu is just the spectral density evaluated at the eigen-frequencies 
    (square root of eigenvalues)
    """
    inds, ω0, d = (lambda u: (u.inds, u.ω0, u.d))(inducing_variable)
    eigen_frequencies = tf.cast(inds+1, gpflow.default_float()) * ω0[None,:]
    S = MaternSpectralDensityND(eigen_frequencies,
                                tf.constant(0.5, dtype=gpflow.default_float()),
                                d, kernel.lengthscales) 
    S = 1/(kernel.variance*tf.clip_by_value(S,gpflow.config.default_jitter(),1E6))
    return Diag(S, is_self_adjoint=True, is_positive_definite=True)

@cov.Kuu.register(LaplacianDirichletFeatures,gpflow.kernels.Matern32)
def Kuu_matern32_ldf(inducing_variable, kernel, jitter=None):
    """
    Kuu is just the spectral density evaluated at the eigen-frequencies 
    (square root of eigenvalues)
    """
    inds, ω0, d = (lambda u: (u.inds, u.ω0, u.d))(inducing_variable)
    eigen_frequencies = tf.cast(inds+1, gpflow.default_float()) * ω0[None,:]
    S = MaternSpectralDensityND(eigen_frequencies,
                                tf.constant(3/2, dtype=gpflow.default_float()),
                                d, kernel.lengthscales)
    S = 1/(kernel.variance*tf.clip_by_value(S,gpflow.config.default_jitter(),1E6))
    return Diag(S, is_self_adjoint=True, is_positive_definite=True)

@cov.Kuu.register(LaplacianDirichletFeatures, gpflow.kernels.Matern52)
def Kuu_matern52_ldf(inducing_variable, kernel, jitter=None):
    """
    Kuu is just the spectral density evaluated at the eigen-frequencies 
    (square root of eigenvalues)
    """
    inds, ω0, d = (lambda u: (u.inds, u.ω0, u.d))(inducing_variable)
    eigen_frequencies = tf.cast(inds+1, gpflow.default_float()) * ω0[None,:]
    S = MaternSpectralDensityND(eigen_frequencies,
                                tf.constant(5/2, dtype=gpflow.default_float()),
                                d, kernel.lengthscales)
    S = 1/(kernel.variance*tf.clip_by_value(S,gpflow.config.default_jitter(),1E6))
    return Diag(S, is_self_adjoint=True, is_positive_definite=True)

@cov.Kuu.register(LaplacianDirichletFeatures, SeparableMatern12)
def Kuu_sep_matern12_ldf(inducing_variable, kernel, jitter=None):
    """
    Kuu is just the spectral density evaluated at the eigen-frequencies 
    (square root of eigenvalues)
    """
    inds, ω0, d = (lambda u: (u.inds, u.ω0, u.d))(inducing_variable)
    eigen_frequencies = tf.cast(inds+1, gpflow.default_float()) * ω0[None,:]
    S = SeparableMaternSpectralDensityND(eigen_frequencies,
                                tf.constant(0.5, dtype=gpflow.default_float()),
                                d, kernel.lengthscales)
    S = 1/(kernel.variance*tf.clip_by_value(S,gpflow.config.default_jitter(),1E6))
    return Diag(S, is_self_adjoint=True, is_positive_definite=True)

@cov.Kuu.register(LaplacianDirichletFeatures, SeparableMatern32)
def Kuu_sep_matern32_ldf(inducing_variable, kernel, jitter=None):
    """
    Kuu is just the spectral density evaluated at the eigen-frequencies 
    (square root of eigenvalues)
    """
    inds, ω0, d = (lambda u: (u.inds, u.ω0, u.d))(inducing_variable)
    eigen_frequencies = tf.cast(inds+1, gpflow.default_float()) * ω0[None,:]
    S = MSeparableMaternSpectralDensityND(eigen_frequencies,
                                tf.constant(3/2, dtype=gpflow.default_float()),
                                d, kernel.lengthscales)
    S = 1/(kernel.variance*tf.clip_by_value(S,gpflow.config.default_jitter(),1E6))
    return Diag(S, is_self_adjoint=True, is_positive_definite=True)

@cov.Kuu.register(LaplacianDirichletFeatures, SeparableMatern52)
def Kuu_sep_matern52_ldf(inducing_variable, kernel, jitter=None):
    """
    Kuu is just the spectral density evaluated at the eigen-frequencies 
    (square root of eigenvalues)
    """
    inds, ω0, d = (lambda u: (u.inds, u.ω0, u.d))(inducing_variable)
    eigen_frequencies = tf.cast(inds+1, gpflow.default_float()) * ω0[None,:]
    S = SeparableMaternSpectralDensityND(eigen_frequencies,
                                tf.constant(5/2, dtype=gpflow.default_float()),
                                d, kernel.lengthscales)
    S = 1/(kernel.variance*tf.clip_by_value(S,gpflow.config.default_jitter(),1E6))
    return Diag(S, is_self_adjoint=True, is_positive_definite=True)

@cov.Kuu.register(LaplacianDirichletFeatures, gpflow.kernels.RBF)
def Kuu_rbf_ldf(inducing_variable, kernel, jitter=None):
    """
    Kuu is just the spectral density evaluated at the eigen-frequencies 
    (square root of eigenvalues)
    """
    inds, ω0, d = (lambda u: (u.inds, u.ω0, u.d))(inducing_variable)
    eigen_frequencies = tf.cast(inds+1, gpflow.default_float()) * ω0[None,:]
    S = RBFSpectralDensityND(eigen_frequencies, kernel.lengthscales)
    S = 1/(kernel.variance*tf.clip_by_value(S,gpflow.config.default_jitter(),1E6))
    return Diag(S, is_self_adjoint=True, is_positive_definite=True)
# -


