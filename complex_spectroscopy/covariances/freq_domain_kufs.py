import tensorflow as tf
import numpy as np
import gpflow
from ..inducing_variables.laplacian_dirichlet_features import LaplacianDirichletFeatures

from gpflow.base import TensorLike
# from gpflow import covariances as cov
from gpflow.config import default_float
from .dispatch import FreqDomainKuf
from ..kernels import *

__all__ = []
# +


def phi_freq_real(w, L, ks, a, w0):
    """The real part of the laplacian dirichlet eigenfunction fourier transformed after
    gaussian windowing to the frequency domain.

    w: evaluation frequencies NxD
    L: length parameter of eigenfunction
    ks: integer eigen numbers to evaluate at MxD
    a: width of gaussian window
    w0: starting eigen-frequency

    """
    pi = gpflow.utilities.to_default_float(np.pi)
    ks = tf.cast(ks, gpflow.default_float())
    L = tf.cast(L, gpflow.default_float()) # length D
    a = tf.cast(a, gpflow.default_float())
    w0 = tf.cast(w0, gpflow.default_float()) # length D
    ksw0 = tf.expand_dims(ks,-3)*tf.expand_dims(tf.expand_dims(w0,-2),-3) # ...x1xMxD
    we = tf.expand_dims(w,-2) # ...xNx1xD
    Le = tf.expand_dims(tf.expand_dims(L,-2),-3)
    # r has shape NxMxD
    r = (tf.sqrt(pi/2.)*a*tf.math.sin(Le*ksw0))*tf.math.exp(-(a**2*tf.square(-we + ksw0))/2.) - \
        (tf.sqrt(pi/2.)*a*tf.math.sin(Le*ksw0))*tf.math.exp(-(a**2*tf.square(we + ksw0))/2.)
    return tf.math.reduce_prod(r, -1) # NxM


def phi_freq_imag(w, L, ks, a, w0):
    """The imaginary part of the laplacian dirichlet eigenfunction fourier transformed after
    gaussian windowing to the frequency domain"""
    pi = gpflow.utilities.to_default_float(np.pi)
    ks = tf.cast(ks, gpflow.default_float())
    L = tf.cast(L, gpflow.default_float())
    a = tf.cast(a, gpflow.default_float())
    w0 = tf.cast(w0, gpflow.default_float())
    ksw0 = tf.expand_dims(ks, -3) * tf.expand_dims(tf.expand_dims(w0, -2), -3)  # ...x1xMxD
    we = tf.expand_dims(w, -2)  # ...xNx1xD
    Le = tf.expand_dims(tf.expand_dims(L, -2), -3)
    r = (tf.sqrt(pi/2.)*a*tf.math.cos(Le*ksw0))*tf.math.exp(-(tf.square(a)*tf.square(we + ksw0))/2.) - \
        (tf.sqrt(pi/2.)*a*tf.math.cos(L*ksw0))*tf.math.exp(-(tf.square(a)*tf.square(-we + ksw0))/2.)
    return tf.math.reduce_prod(r, -1) # NxM



@FreqDomainKuf.register(LaplacianDirichletFeatures, gpflow.kernels.Matern12, TensorLike)
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

@FreqDomainKuf.register(LaplacianDirichletFeatures, gpflow.kernels.Matern32, TensorLike)
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

@FreqDomainKuf.register(LaplacianDirichletFeatures, gpflow.kernels.Matern52, TensorLike)
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

@FreqDomainKuf.register(LaplacianDirichletFeatures, SeparableMatern12, TensorLike)
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

@FreqDomainKuf.register(LaplacianDirichletFeatures, SeparableMatern32, TensorLike)
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

@FreqDomainKuf.register(LaplacianDirichletFeatures, SeparableMatern52, TensorLike)
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

@FreqDomainKuf.register(LaplacianDirichletFeatures, gpflow.kernels.RBF, TensorLike)
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

