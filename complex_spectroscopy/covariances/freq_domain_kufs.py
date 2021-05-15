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

def causal_gaussian_window_nonsep(f, a):
    """
    f: evaluation frequencies ...MxNxD,
    a: width of gaussian window, 1x1xD
    """
    pi = tf.constant(np.pi, dtype=f.dtype)
    factor = 2/tf.sqrt(pi)
    imag_part = factor*tf.math.special.dawsn(f*a)
    real_part = tf.math.exp(-tf.square(a*f))
    r = tf.complex(real=real_part, imag=imag_part)
    return r #...MxN

def gaussian_window_nonsep(f, a):
    """
    f: evaluation frequencies ...MxNxD,
    a: width of gaussian window, 1x1xD
    """
    real_part = tf.math.exp(-tf.square(a*f))
    imag_part = tf.zeros_like(real_part)
    r = tf.complex(real=real_part, imag=imag_part)
    return r #...MxN

def causal_sinc_window(w, a, L):
    sinc = tf.experimental.numpy.sinc
    real_part = (L/a)*tf.cos(L*w/2)*sinc(L*w/(2*a*np.pi))
    imag_part = (L/a)*tf.sin(L*w/2)*sinc(L*w/(2*a*np.pi))
    return tf.complex(real=real_part, imag=imag_part)

def sinc_window(w, a, L):
    sinc = tf.experimental.numpy.sinc
    real_part = (L/a)*sinc(L*w/(2*a*np.pi))
    return tf.complex(real=real_part, imag=tf.zeros_like(real_part))


def phi_freq(w, L, ks, alpha, window_function = sinc_window):
    """The laplacian dirichlet eigenfunction fourier transformed after
    gaussian windowing to the frequency domain.

    w: evaluation angular frequencies NxD
    L: length parameter of eigenfunction
    ks: integer eigen numbers to evaluate at MxD
    a: width of gaussian window in angular frequency

    in summary, this evaluates to:

    2*pi*i*exp(-k*pi*i/2)*window_fun(k*pi/L+2*w)/sqrt(L) +
    -2*pi*i*exp(k*pi*i/2)*window_fun(k*pi/L-2*w)/sqrt(L)
    """
    M = tf.cast(tf.shape(ks)[-2],w.dtype)
    pi = gpflow.utilities.to_default_float(np.pi)
    ks = tf.cast(ks, gpflow.default_float())
    kse = tf.expand_dims(ks, -2)  # ...Mx1xD
    ae = tf.expand_dims(tf.expand_dims(alpha,-2),-3)  # ...1x1xD
    Le = tf.expand_dims(tf.expand_dims(L,-2),-3)  # ...1x1xD
    we = tf.expand_dims(w, -3)  # ...x1xNxD
    arg1 = kse*pi/Le + 2*we  # ...xMxNxD
    arg2 = kse*pi/Le - 2*we  # ...xMxNxD
    factor0 = tf.complex(real=tf.zeros_like(pi), imag=pi*2.)
    factorarg = tf.complex(real=tf.zeros_like(kse), imag=0.5*np.pi*kse)  # ikpi/2
    factor1 = factor0*tf.math.exp(-factorarg)/tf.complex(real=tf.math.sqrt(Le), imag=tf.zeros_like(Le))
    factor2 = -factor0*tf.math.exp(factorarg)/tf.complex(real=tf.math.sqrt(Le), imag=tf.zeros_like(Le))
    wf1 = factor1*window_function(arg1, ae, L)
    wf2 = factor2*window_function(arg2, ae, L)
    return tf.math.reduce_prod(wf1+wf2, -1)  #...xMxN


def general_kuf(inducing_variable, kernel, X, alpha):
    inds, d, L = (lambda u: (u.inds, u.d, u.L))(inducing_variable)
    if alpha is None:
        alpha = L
    return phi_freq(X, L, inds, alpha)

@FreqDomainKuf.register(LaplacianDirichletFeatures, gpflow.kernels.Matern12, TensorLike)
def Kuf_matern12_ldf_time_domain(inducing_variable, kernel, X, alpha=None):
    return general_kuf(inducing_variable, kernel, X, alpha=alpha)

@FreqDomainKuf.register(LaplacianDirichletFeatures, gpflow.kernels.Matern32, TensorLike)
def Kuf_matern32_ldf_time_domain(inducing_variable, kernel, X, alpha=None):
    return general_kuf(inducing_variable, kernel, X, alpha=alpha)

@FreqDomainKuf.register(LaplacianDirichletFeatures, gpflow.kernels.Matern52, TensorLike)
def Kuf_matern52_ldf_time_domain(inducing_variable, kernel, X, alpha=None):
    return general_kuf(inducing_variable, kernel, X, alpha=alpha)

@FreqDomainKuf.register(LaplacianDirichletFeatures, SeparableMatern12, TensorLike)
def Kuf_sep_matern12_ldf_time_domain(inducing_variable, kernel, X, alpha=None):
    return general_kuf(inducing_variable, kernel, X, alpha=alpha)

@FreqDomainKuf.register(LaplacianDirichletFeatures, SeparableMatern32, TensorLike)
def Kuf_sep_matern32_ldf_time_domain(inducing_variable, kernel, X, alpha=None):
    return general_kuf(inducing_variable, kernel, X, alpha=alpha)

@FreqDomainKuf.register(LaplacianDirichletFeatures, SeparableMatern52, TensorLike)
def Kuf_sep_matern52_ldf_time_domain(inducing_variable, kernel, X, alpha=None):
    return general_kuf(inducing_variable, kernel, X, alpha=alpha)

@FreqDomainKuf.register(LaplacianDirichletFeatures, gpflow.kernels.RBF, TensorLike)
def Kuf_RBF_ldf_time_domain(inducing_variable, kernel, X, alpha=None):
    return general_kuf(inducing_variable, kernel, X, alpha=alpha)

