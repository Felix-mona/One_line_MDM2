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



def len_btwn_points(a, b):
    ab = b - a
    len = np.sqrt(abs(ab[:,:1] + ab[:,1:]))
    return len


def calculate_distances(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """
    Calculate the Euclidean distance between pairs of (x, y) coordinates.

    Args:
        coords1: NumPy array of shape (N, 2).
        coords2: NumPy array of shape (N, 2).

    Returns:
        NumPy array of shape (N,) containing the distances.
    """
    return np.linalg.norm(coords1 - coords2, axis=1)



def calc_metric(new_image, old_image, mtype=None):

    if mtype == 'accuracy' or mtype == 'Accuracy':

        A = np.column_stack((
            np.real(old_image),
            np.imag(old_image)
        ))

        B = np.column_stack((
            np.real(new_image),
            np.imag(new_image)
        ))

        # Calculate each directed Hausdorff distance
        d1, i1, j1 = directed_hausdorff(A, B)
        d2, i2, j2 = directed_hausdorff(B, A)

        # Determine which direction produces the Hausdorff distance
        if d1 >= d2:
            d = d1
            point1 = i1  # Original image
            point2 = j1  # Nearest reconstructed point
            direction = 'original_to_reconstructed'

        else:
            d = d2
            point1 = j2  # Nearest original point
            point2 = i2  # Reconstructed image
            direction = 'reconstructed_to_original'

        # Normalise by image scale
        both = np.concatenate((old_image, new_image))
        scale = np.hypot(
            np.ptp(both.real),
            np.ptp(both.imag)
        )

        accuracy = d / scale if scale > 0 else 0.0

        index = (point1, point2, direction)

        return accuracy, index

    elif mtype == 'smoothness' or mtype == 'Smoothness':

        A_coords = np.column_stack((
            np.real(new_image),
            np.imag(new_image)
        ))

        B_coords = np.roll(A_coords, 1, axis=0)
        C_coords = np.roll(A_coords, -1, axis=0)

        c = calculate_distances(A_coords, B_coords)
        b = calculate_distances(A_coords, C_coords)
        a = calculate_distances(B_coords, C_coords)

        cos_angle = (b**2 + c**2 - a**2) / (2*b*c)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        angle = np.arccos(cos_angle)

        return 1 - np.mean(angle/np.pi)

    else:
        pass











    
