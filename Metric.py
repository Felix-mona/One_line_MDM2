"""
This contains the functions related to the producing and comparing of the different Loss functions that we will produce
We could now use different metrics to show for different things:
 - informative ( Does the sketch match the original? )
 - Fluid ( Does the curve flow? )
 - Clean ( Is there junk in the sketch )
 - Simplicity ( Is it the smallest cutoff K possible )
 - Artefacts ( How far the line strays outside the original bounding box, Gibb's ringing.)

"""
import numpy as np
import matplotlib.pyplot as plt
import Fourier_reconsctruction as fourier
from scipy.spatial.distance import directed_hausdorff

def Loss(accuracy,smoothness,alpha=1,beta=1):
    """
    INPUT:
     - alpha, the coefficient of the accuracy
     - beta, the coefficient of the smoothness


    OUTPUT
     - loss, the Loss value of the current image
    If no alpha and Beta are given then the default values will be equal and the Loss(x) will purely depend on the smoothness and accuracy
    """

    #error checking that alpha and beta == 1
    if alpha + beta != 1:
        print("error, alpha plus beta must equal 1")
        return

    #error checking the accuracy and smoothness to be between bounds and normalised
    if (accuracy < 0  or accuracy > 1) or (smoothness < 0 or smoothness > 1):
        print("the accuracy or the smoothness is out of bounds, should be between [0,1]") 

    loss = alpha*accuracy + beta*smoothness
    return loss

def gather_lowest_metric(coefficients,max_frequency_cutoff,samples,alpha_beta_pairs):
    """
    INPUT:
     - coefficients, a list of the fourier coefficients
     - max frequency - integer, The highest frequency that we wish to sample the image with
     - samples - integer, the amount of different frequencies cutoffs to sample, linearly from 0 to max_frequency (Might have something to do with the discretisation number initially!!)
     - alpha_beta pairs - list of tuples, a list containing coordinate style alpha and beta pairs to test through the different frequencies

     OUTPUT:
      - [k, alpha, beta] the 3 values for which the loss function scored the lowest
    """
    frequencies = np.linspace(1,max_frequency_cutoff,samples)
    combination_per_freq = []
    
    for k in frequencies:
        k = int(k) #cast to int as the frequencies are floats and we need to use them as integers for the filter function
        min_loss = 10 # make a starting min loss above the highest possible loss
        minimum_combination = []
        kfiltered = fourier.filter_coeffs(coefficients,k)

        for alpha, beta in alpha_beta_pairs:
            #compute metric values and normalise from the metric file
            temp_ifd = fourier.inverse_fourier_descriptors(kfiltered)
            temp_accuracy = calc_metric(temp_ifd,coefficients,'accuracy')
            temp_smoothness = calc_metric(temp_ifd,coefficients,'smoothness')

            temp_loss = Loss(temp_accuracy,temp_smoothness,alpha=alpha, beta=beta)
            if (temp_loss) < min_loss:
                min_loss = temp_loss
                minimum_combination = [k,alpha,beta] #kth frequency, current alpha, current beta E.G [50,0,1]
            else:
                continue

        combination_per_freq.append(minimum_combination)

    return combination_per_freq



def len_btwn_points(A, B):
    AB = (A - B)**2
    len = np.sqrt(abs(AB[:,:1] + AB[:,1:]))
    return len


def calc_metric(new_image,old_image,mtype=None):
    """
    Calculates the hausdorf distance between the original image and the redrawn but with restricted frequencies image
    INPUT:
     - new_image, array of the new complex coords of the redrawn image
     - old_image, array of the complex coords from the original image
     - mtype, string between 'accuracy' or 'smoothness'

    OUTPUT:
     - float, [0,1] the accuracy rating of the redrawn image
    """
    
    if mtype == 'accuracy' or mtype == 'Accuracy':
        """
        accuracy branch
        Symmetric Hausdorff distance between two sets of complex points.

        old_points, new_points : 1D arrays of complex numbers (can be different lengths)

        Returns a float: the largest distance from any point in one set
        to its nearest neighbour in the other set.
        """
        A = np.column_stack((np.real(old_image), np.imag(old_image)))
        B = np.column_stack((np.real(new_image), np.imag(new_image)))

        index = [(directed_hausdorff(A, B)[1],directed_hausdorff(A, B)[2]), (directed_hausdorff(B, A)[1], directed_hausdorff(B, A)[2])] # the index of the point in the original image that is furthest away from the new image and vice versa
        d = max(directed_hausdorff(A, B)[0], directed_hausdorff(B, A)[0])# symmetric hausdorf distance between the two sets of points

        both = np.concatenate((old_image, new_image))
        scale = np.hypot(np.ptp(both.real), np.ptp(both.imag))
        
        return d / scale if scale > 0 else 0.0,index # normalise the distance by the scale of the image, if the scale is 0 then return 0.0
    
    elif mtype == 'smoothness' or mtype == 'Smoothness':
        """
        smoothness branch
        """
        A_coords = np.column_stack((np.real(new_image)), np.imag(new_image))  # makes a list of x,y coords from the new image
        B_coords = np.delete(np.vstack([A_coords[-1],A_coords]), (-1), axis=0) # shifts the list one along the path and one backwards along the path
        C_coords = np.delete(np.vstack([A_coords,A_coords[0]]), (0), axis=0)

        c = len_btwn_points(A_coords,B_coords) # calculates three distances between the three current points
        b = len_btwn_points(A_coords,C_coords)
        a = len_btwn_points(B_coords,B_coords)

        angle = np.arcos((b**2 + c**2 - a**2) / (2*b*c)) # takes the cosine rule between the points to find the middle angle

        return np.mean(angle/np.pi)  # returns the mean angle of all the different points within the list.

    else:
        pass










    
