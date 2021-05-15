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

__all__ = ['dtype_to_ctype']


def dtype_to_ctype(dtype):
    if dtype is tf.float64 or dtype is np.float64:
        ctype = tf.complex128
    elif dtype is tf.float32 or dtype is np.float32:
        ctype = tf.complex64
    else:
        raise ValueError('incompatible dtype requested, must be either float32 or float64')
    return ctype
