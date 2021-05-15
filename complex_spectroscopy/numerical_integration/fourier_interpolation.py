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

import numpy as np
import tensorflow as tf
from .util import dtype_to_ctype
from .resampling import re_sample_1D_complex_functions
from gpflow import default_float

__all__ = [
           'BatchedFourierAxis',
           'batched_resampled_1DFFT',
           'batched_shifted_resampled_1DFFT', 
           'batched_fourier_interp1D',
           'batched_shifted_normalized_resampled_1DFFT',
           'batched_normalized_fourier_interp1D'
          ]


class BatchedFourierAxis(object):
    """
    args:
    sample_lb: a J length tensor giving the lower bounds for J functions
    sample_ub: a J length tensor givine the upper bounds for J functions
    N: an integer number of points for the sample and Fourier axis
    
    returns:
    an object holding various parameters needed to construct a Fourier axis. 
    Of note: 
        * w0 the spacing of the frequency axis (length J)
        * k a properly ordered integer sequence to match the FFT. Using these sequences
          the frequency axis may be constructed by w0[:,None]*k[None,:]. Handles even
          and odd numbered values of N correctly.
        * w the frequency axis evaluated as described above for convenience (JxN)
        * x the sample axis (JxN)
        * L the half range of the sample axis (length J)
        * delta_x the spacing of the sample axis (length J)
        * offset_lb the lower bound of the sample axis minus delta_x, which is useful
            in interpolation.
    """
    
    def __init__(self, sample_lb, sample_ub, N):
        dtype = default_float()
        if N % 2 > 0:  # odd case
            # [0, 1, ..., (N-1)/2, -(N-1)/2, ..., -1]
            kha = tf.cast(tf.range((N-1)//2+1),dtype)
            khb = tf.cast(tf.range(1,(N-1)//2+1),dtype)
        else:  # even case
            # [0, 1, ..., N/2-1, -N/2, ..., -1]
            kha = tf.cast(tf.range(N//2),dtype)
            khb = tf.cast(tf.range(1,N//2+1),dtype)
        sample_lb = np.atleast_1d(sample_lb)
        sample_ub = np.atleast_1d(sample_ub)
        R = tf.convert_to_tensor(sample_ub - sample_lb,dtype=dtype)  # J length
        self._L = R/2  # J length
        self._delta_x = R/(N-1)  # J length
        self._x = tf.cast(tf.range(N),dtype)[None,:]*self._delta_x[:,None]+sample_lb[:,None]  # JxN
        self._offset_lb = sample_lb-self._delta_x
        self._k = tf.concat([kha,-khb[::-1]],0)
        self._w0 = 1/(self._delta_x*N)  # J length
        self._w = self._k[None,:]*self._w0[:,None]  # JxN
        self._lb = sample_lb
        self._ub = sample_ub
        self._N = N
    
    @property
    def w0(self):
        return self._w0
    
    @property
    def k(self):
        return self._k
    
    @property
    def delta_x(self):
        return self._delta_x
    
    @property
    def L(self):
        return self._L
    
    @property
    def x(self):
        return self._x
    
    @property
    def w(self):
        return self._w
    
    @property
    def offset_lb(self):
        return self._offset_lb
    
    @property
    def lb(self):
        return self._lb
    
    @property
    def ub(self):
        return self._ub
    
    @property
    def N(self):
        return self._N


def batched_resampled_1DFFT(new_lb, new_ub, N, samples, old_lb, old_ub):
    """
    args:
    samples: complex typed samples of a 1D function, presumed to be taken on old_lb to old_ub in M evenly
            spaced steps. Samples must be of size JxBxM.
    old_lb: a 1D tensor of floats J in length that correspond to the lower bounds the sampled functions were
            evaluated on. 
    old_ub: a 1D tensor of floats J in length that correspond to the upper bounds the sampled functions were
            evaluated on.
    new_lb: a 1D tensor of floats J in length that correspond to the lower bounds the sampled functions will
            be evaluated on.
    new_ub: a 1D tensor of floats J in length that correspond to the upper bounds the sampled functions will
            be evaluated on.
    N: an integer number of new sample points to evaluate the function on. Resampling is done 
            via linear interpolation, where function values outside the range old_lb to old_ub
            are set to zero.
            
    returns: the FFT coefficients of the resampled, batched 1D function of size JxBxN
    """
    dtype = default_float()
    ctype = dtype_to_ctype(dtype)
    new_axis = tf.linalg.matrix_transpose(tf.cast(tf.linspace(new_lb, new_ub, N),dtype))  # JxN
    rsamples = re_sample_1D_complex_functions(new_axis, samples, old_lb, old_ub)  # JxBxN
    N = tf.cast(tf.shape(rsamples)[-1],ctype)
    return tf.signal.fft(rsamples)/N  # transforms over -1 dim


def batched_shifted_resampled_1DFFT(new_lb, new_ub, N, samples, old_lb, old_ub):
    """
    args:
    samples: complex typed samples of a 1D function, presumed to be taken on old_lb to old_ub in M evenly
            spaced steps. Samples must be of size JxBxM.
    old_lb: a 1D tensor of floats J in length that correspond to the lower bounds the sampled functions were
            evaluated on. 
    old_ub: a 1D tensor of floats J in length that correspond to the upper bounds the sampled functions were
            evaluated on.
    new_lb: a 1D tensor of floats J in length that correspond to the lower bounds the sampled functions will
            be evaluated on.
    new_ub: a 1D tensor of floats J in length that correspond to the upper bounds the sampled functions will
            be evaluated on.
    N: an integer number of new sample points to evaluate the function on. Resampling is done 
            via linear interpolation, where function values outside the range old_lb to old_ub
            are set to zero.
            
    returns: the shifted FFT coefficients of the resampled, batched 1D function of size JxBxN
    
    Note: this function is different from batch_resampled_1DFFT in that the coefficients it returns are
    phase shifted so that interpolation can proceed using unshifted basis functions.
    """
    iaxis = BatchedFourierAxis(new_lb, new_ub, N)
    ctype = dtype_to_ctype(default_float())
    shifts = tf.exp(1j*tf.cast(2.*np.pi*iaxis.w*(-iaxis.lb[:,None]), ctype))  # JxN
    An = batched_resampled_1DFFT(iaxis.lb, iaxis.ub, iaxis.N, samples, old_lb, old_ub)  # JxBxN
    Zn = An*shifts[:,None,:]  # JxBxN
    return Zn  # transforms over -1 dim


def batched_shifted_normalized_resampled_1DFFT(new_lb, new_ub, N, samples, old_lb, old_ub):
    """
    args:
    samples: complex typed samples of a 1D function, presumed to be taken on old_lb to old_ub in M evenly
            spaced steps. Samples must be of size JxBxM.
    old_lb: a 1D tensor of floats J in length that correspond to the lower bounds the sampled functions were
            evaluated on. 
    old_ub: a 1D tensor of floats J in length that correspond to the upper bounds the sampled functions were
            evaluated on.
    new_lb: a 1D tensor of floats J in length that correspond to the lower bounds the sampled functions will
            be evaluated on.
    new_ub: a 1D tensor of floats J in length that correspond to the upper bounds the sampled functions will
            be evaluated on.
    N: an integer number of new sample points to evaluate the function on. Resampling is done 
            via linear interpolation, where function values outside the range old_lb to old_ub
            are set to zero.
            
    returns: the shifted FFT coefficients of the resampled, batched 1D function of size JxBxN
    
    Note: this function is different from batch_resampled_1DFFT in that the coefficients are phase shifted
    so that interpolation can be done using unshifted basis functions. Furthermore, we normalize the
    shifts so that integration on old_lb to old_ub is easier.
    
    Because of all the extra transformation involved, we return the frequency axis scaling factor L and shift
    """
    L = 0.5*(old_ub - old_lb)
    mid_point = 0.5*(old_ub + old_lb)
    transformed_new_lb = (new_lb - mid_point)/L
    transformed_new_ub = (new_ub - mid_point)/L
    iaxis = BatchedFourierAxis(transformed_new_lb, transformed_new_ub, N)
    ctype = dtype_to_ctype(default_float())
    shifts = tf.exp(1j*tf.cast(2.*np.pi*iaxis.w*(-iaxis.lb[:,None]), ctype))  # JxN
    Zn = batched_resampled_1DFFT(iaxis.lb, iaxis.ub, iaxis.N, samples,
                               -tf.ones_like(old_lb), tf.ones_like(old_ub))  # JxBxN
    An = Zn*shifts[:,None,:]  # JxBxN
    return An, iaxis.w, L, mid_point


def batched_fourier_interp1D(x, i_lb, i_ub, iN, samples, sample_axis_lb, sample_axis_ub):
    """
    Constructs an N point interpolant from M samples. Interpolant can have a different lower_bound/upper_bound
    than the samples passed in. Interpolant is than evaluated on x, which may be arbitrarily spaced.
    
    args:
    x: new axis (not necessarily evenly spaced) on which to evaluate the function.
    i_lb: a 1D tensor of floats J in length that correspond to the lower bounds the sampled functions will
            be interpolated with.
    i_ub: a 1D tensor of floats J in length that correspond to the upper bounds the sampled functions will
            be interpolated with.
    i_N: the number of iterpolation basis functions.
    samples: complex typed samples of a 1D function, presumed to be taken on -1 to 1 in M evenly spaced
            steps. Samples must be of size JxBxM.
    sample_axis_lb: a 1D tensor of floats J in length that correspond to the lower bounds the sampled
                    functions were evaluated on. 
    sample_axis_ub: a 1D tensor of floats J in length that correspond to the upper bounds the sampled
                    functions were evaluated on.
            
    returns: an interpolated version of the sampled functions evalauted at x
    """
    iaxis = BatchedFourierAxis(i_lb, i_ub, iN)
    ctype = dtype_to_ctype(default_float())
    Zn = batched_shifted_resampled_1DFFT(i_lb, i_ub, iN, samples, sample_axis_lb, sample_axis_ub)
    B = tf.exp(1j*tf.cast(2*np.pi*iaxis.w[:,:,None]*x[:,None,:],ctype))  # JxNxQ
    return tf.einsum('ijk,ikm->ijm',Zn,B)


def batched_normalized_fourier_interp1D(x, i_lb, i_ub, iN, samples, sample_axis_lb, sample_axis_ub):

    """
    Constructs an N point interpolant from M samples. Interpolant can have a different lower_bound/upper_bound
    than the samples passed in. Interpolant is than evaluated on x, which may be arbitrarily spaced.
    
    args:
    x: new axis (not necessarily evenly spaced) on which to evaluate the function.
    i_lb: a 1D tensor of floats J in length that correspond to the lower bounds the sampled functions will
            be interpolated with.
    i_ub: a 1D tensor of floats J in length that correspond to the upper bounds the sampled functions will
            be interpolated with.
    i_N: the number of iterpolation basis functions.
    samples: complex typed samples of a 1D function, presumed to be taken on -1 to 1 in M evenly spaced
            steps. Samples must be of size JxBxM.
    sample_axis_lb: a 1D tensor of floats J in length that correspond to the lower bounds the sampled
                    functions were evaluated on. 
    sample_axis_ub: a 1D tensor of floats J in length that correspond to the upper bounds the sampled
                    functions were evaluated on.
            
    returns: an interpolated version of the sampled functions evalauted at x
    
    note: internally, this uses a normalized coordinate system for the samples. This function is mainly
    for debugging batched_shifted_normalized_resampled_1DFFT
    """
    ctype = dtype_to_ctype(default_float())
    An, ξ, L, mid_point = batched_shifted_normalized_resampled_1DFFT(i_lb, i_ub, iN,
                                                  samples, sample_axis_lb,
                                                  sample_axis_ub)
    x_transformed = (x[:,None,:]-mid_point[:,None,None])/L[:,None,None]  # Jx1xQ
    B = tf.exp(1j*tf.cast(2*np.pi*ξ[:,:,None]*x_transformed,ctype))  # JxNxQ
    return tf.einsum('ijk,ikm->ijm',An,B)


def fourier_interp1D_given_coeffs(x, An, ξ, L, mid_point):

    """
    given coefficients, interpolate on x
    """
    ctype = dtype_to_ctype(default_float())
    x_transformed = (x[:,None,:]-mid_point[:,None,None])/L[:,None,None]  # Jx1xQ
    B = tf.exp(1j*tf.cast(2*np.pi*ξ[:,:,None]*x_transformed,ctype))  # JxNxQ
    return tf.einsum('ijk,ikm->ijm',An,B)
