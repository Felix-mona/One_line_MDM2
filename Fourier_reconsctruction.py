"""
Rough outline of the Fourier reconstruction algorithm.
- Take in a SVG file and convert it to a set of points. (s,x) (s,y) where s is the distance along the path and (x,y) are the coordinates of the point.
- sum together the points to get a single complex number for each point. (s, x + iy)
- apply the Fourier transform to the complex numbers to get a set of coefficients.
- apply some sort of noise filter or frequency cut off to the coefficients. -- the main part that will control what the recreated path looks like.
- apply the inverse Fourier transform to the filtered coefficients to get a new set of complex numbers.
- plot new points from the complex numbers to get a new path.
- apply metrics of some sort to compare the new path to the original path and see how well it matches.
"""
import numpy as np
import matplotlib.pyplot as plt
import svgpathtools as svg
import scipy.fft as fft



def forward_fourier(z):
    """"
    Input:
    - z: a list of imaginary numbers representing the path at distance s from start, [(x + yj)]


    Output:
    - coefficients, List of complex numbers representing the coefficients of the fourier transform
    - freq_channels, List of integers that are the whole number frequencies present in the curve.

    """
    n = len(z)
    coefficients = fft.fft(z, norm="forward")
    freq_channels = fft.fftfreq(n, d=1/n).astype(int)
    return coefficients, freq_channels

 
 
def filter_coeffs(coefficients, cutoff=None):
    """
    Input:
    - coefficients: array of complex numbers representing the coefficients of the fourier transform, must be a list where the x[0] is the zero frequency term
    - cutoff: Integer, the cutoff frequency for the filter, highest value it can take is N/2 the sample amount of the curve


    Output:
    - filtered_coefficients: list of complex numbers representing the filtered coefficients of the fourier transform, must be a list where the x[0] is the zero frequency term
    """
    coefficients = np.asarray(coefficients)
    ##loop to handle if there are no params given
    if cutoff is None:
        return coefficients

    n = len(coefficients)
    cutoff = min(int(cutoff), (n-1)//2 ) # trims the highest cut off to be within the N//2 real, so that there will not be any errors, or will use the cutoff value given

    fourier_filtered = coefficients.copy() # make a copy as to not change the original
    
    fourier_filtered[cutoff + 1 : n - cutoff] = 0 # cut offs the frquency bands above the cutoff and below the negative cutoff

    return fourier_filtered


    
 
 
def inverse_fourier_descriptors(coefficients):
    """
    Input:
    - coefficients: list of complex numbers representing the coefficients of the fourier transform, must be a list where the x[0] is the zero frequency term
    
    Output:
    - z: a list of imaginary numbers representing the path at distance s from start, [(x + yj)]
    """
    z = fft.ifft(coefficients, norm="forward") #
    return z


