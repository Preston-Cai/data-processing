"""
Helper functions for PSNR and SSIM functions.
"""

import numpy as np

def l(f, g):
    
    c1 = 0.001  # a small postive constant
    return (2 * mean_luminance(f) * mean_luminance(g) + c1) \
        / (mean_luminance(f) ** 2 + mean_luminance(g) ** 2 + c1)


def c(f, g):
    
    c2 = 0.001
    return (2 * standard_deviation(f) * standard_deviation(g) + c2) \
        / (standard_deviation(f) ** 2 + standard_deviation(g) ** 2 + c2)


def s(f, g):

    c3 = 0.001
    return (covariance(f, g) + c3) \
        / (standard_deviation(f) * standard_deviation(g) + c3)
            

def mean_luminance(f):
    f = np.array(f, dtype=np.float64)

    coefficients = np.array([0.2126, 0.7152, 0.0722])

    if f.ndim == 3 and f.shape[2] == 3:
        luminance = f[..., 0] * coefficients[0] 
        + f[..., 1] * coefficients[1] + f[..., 2] * coefficients[2]

        return np.mean(luminance)
    
    else:
        return 'ValueError: image must be in RGB'
    
def standard_deviation(f: np.array) -> float:
    
    sum_sqr = np.sum((f.flatten() - np.mean(f)) ** 2)
    return np.sqrt(sum_sqr / f.size)


def covariance(f, g) -> float:

    f = np.array(f, dtype=np.float64).flatten()
    g = np.array(g, dtype=np.float64).flatten()
    mean_f = np.mean(f)
    mean_g = np.mean(g)

    cov = np.mean((f - mean_f) * (g - mean_g))
    return cov