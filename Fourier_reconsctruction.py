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

## Creating a list of imgaging points from the SVG file
## getting a list of coordindates in the shape of an elipse and starting at a point
def ellipse(a, b, n):
    """n complex points on the ellipse x²/a² + y²/b² = 1, centred on the origin."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return a * np.cos(t) + 1j * b * np.sin(t)






def fourier_coeffs(z):
    """"
    Input:
    - z: a list of imaginary numbers representing the path at distance s from start, [(x + yj)]
    Output:
    - coefficients, List of complex numbers representing the coefficients of the fourier transform
    - freq_channels, List of integers that are the whole number frequencies present in the curve.

    """
    n = len(z)
    coefficients = fft.fft(z) / n
    freq_channels = fft.fftfreq(n, d=1/n).astype(int)
    return coefficients, freq_channels

 
 
##def filter_coeffs(coefficients, cutoff=None, filter_type=None, filter_order=None):
    """
    Input:
    - coefficients: list of complex numbers representing the coefficients of the fourier transform, must be a list where the x[0] is the zero frequency term
    - cutoff: float, the cutoff frequency for the filter
    - filter_type: str, the type of filter to apply ('lowpass', 'highpass', 'bandpass', 'bandstop')
    - filter_order: int, the order of the filter to apply
    Output:
    - filtered_coefficients: list of complex numbers representing the filtered coefficients of the fourier transform, must be a list where the x[0] is the zero frequency term
    """
    ##loop to handle if there are no params given
    if cutoff is None or filter_type is None or filter_order is None:
        return coefficients

    pass


    
 
 
def reconstruct(coefficients):
    """
    Input:
    - coefficients: list of complex numbers representing the coefficients of the fourier transform, must be a list where the x[0] is the zero frequency term
    Output:
    - z: a list of imaginary numbers representing the path at distance s from start, [(x + yj)]
    """
    z = fft.ifft(coefficients*len(coefficients))
    return z

    
 
def main():
    z = ellipse(4,19,300)

    plt.plot(z.real, z.imag)
    plt.axis('equal')
    plt.show()

    coeffs, freq = fourier_coeffs(z)

    plt.plot(freq, np.abs(coeffs))
    plt.title("Fourier Coefficients")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()

    print("the coefficients of the fourier transform are:", coeffs,
          "\n\n\nThe Frequency channels are:", freq)


main()


