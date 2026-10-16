"""
Rough outline of the Fourier reconstruction algorithm.
- Take in a SVG file and convert it to a set of points. (s,x) (s,y) where s is the distance along the path and (x,y) are the coordinates of the point.
- sum together the points to get a single complex number for each point. (s, x + iy)
- apply the Fourier transform to the complex numbers to get a set of coefficients.
- apply some sort of noise filter or frequency cut off to the coefficients.
- apply the inverse Fourier transform to the filtered coefficients to get a new set of complex numbers.
- plot new points from the complex numbers to get a new path.
- apply metrics of some sort to compare the new path to the original path and see how well it matches.
"""



