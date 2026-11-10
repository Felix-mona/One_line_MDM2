import Metric as met
from SVG_Manipulation import Image
import Fourier_reconsctruction as fourier
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import svgwrite


"""
USAGE OF MAIN.py:
- has some routines in it that will plot our reconstructed curves differently, will have to read and understand what is going on
before just pressing play, some code should commented out otherwise its all a mess and will be saved to a random place.

Accuracy - the hausdorf distance, between x and y (0,1) euclidian percentage distance between the original and new image, where 0 is perfect and 1 is the worst case scenario.
Fixed decided accuracies - 0.002,0.005,0.05,0.1,0.2
"""


FILEPATH = r"C:\Users\felix\Uni\Year 2\MDM2\Primitive_line_project\Project_Files\One_line_MDM2\Images\Indonesia\Indonesia_output_polyline.svg"
N_SAMPLES = 1000
ACCURACIES = [(0.002),
              (0.005),
              (0.05),
              (0.1),
              (0.2)]

image_1 = Image(str(FILEPATH),N_SAMPLES)
image_1.sample_paths() # fills the object with the complex coords

coeffs, freqs = fourier.forward_fourier(image_1.points) #gathers complex coefficients and frequencies from the original image
best_k = [] # initialised to be filled with the best frequency cutoffs for each accuracy threshold
best_k_indexes = [] # holds the coords where the best k is found, for plotting the shortest distance between the two curves


# basic loop that increases the frequency cutoff by 1 until the accuracy threshold is reached, or the maximum frequency cutoff is reached. The best frequency cutoff for each accuracy threshold is stored in best_k
for accuracy_thresh in ACCURACIES:
    max_pointer = int(N_SAMPLES/2)
    pointer_k = 1 #frequency cutoff to start with, will be changed in the while loop
    pointer_accuracy = 10

    while pointer_accuracy > accuracy_thresh:
        filtered = fourier.filter_coeffs(coeffs,pointer_k)
        inverse = fourier.inverse_fourier_descriptors(filtered)
        pointer_accuracy,index = met.calc_metric(inverse,image_1.points,mtype='Accuracy')

        if pointer_accuracy < accuracy_thresh: # case where the accuracy has been reached
            break
        elif pointer_k >= max_pointer: # case where the frequency cutoff has reached the maximum and the accuracy has not been reached
            print("max frequency cutoff reached, accuracy not reached")
            pointer_k = -1 # set the pointer_k to -1 to indicate that the accuracy was not reached
            break
        else:
            pointer_k += 1 # increment the pointer_k to increase the frequency cutoff and get a better accuracy

    best_k.append(pointer_k)
    best_k_indexes.append(index)
    print("Working...")

#plotting loop that will also write the images to a file, use no padding and large font size otherwise hard to see on overleaf
# plots over the 5 different accuracy thresholds, change range to 1 and comment out the new image related code to get the original plotted out
# comment out lines that you do not need
for i in range(5):
    complex_coords = fourier.inverse_fourier_descriptors(fourier.filter_coeffs(coeffs, best_k[i]))
    longest_distance = [(np.real(image_1.points[index[0][0]]),np.imag(image_1.points[index[0][0]])),(np.real(inverse[index[1][0]]),np.imag(inverse[index[1][0]]))] 
    x,y = np.real(complex_coords), np.imag(complex_coords)

    # Plotting the new image over the original image
    plt.plot(x,y,label=f'{best_k[i]} K cutoff')
    plt.plot(np.real(image_1.points),np.imag(image_1.points),label=f'Original Image')

    # plots the points where the hausdorf distance was taken from
    plt.plot(np.real(image_1.points[index[0][0]]),np.imag(image_1.points[index[0][0]]),'ro',label='Furthest point from new image')
    plt.plot(np.real(inverse[index[1][0]]),np.imag(inverse[index[1][0]]),'go',label='Furthest point from original image')

    #plots the line between the two points for the hausdorf distance
    plt.plot([longest_distance[0][0],longest_distance[1][0]],[longest_distance[0][1],longest_distance[1][1]],'k--',label='Longest distance between points') 

    plt.title(f'Indonesia {i}', fontsize=20)
    plt.axis('off')
    plt.legend()

    #saves the plot created within the current directory, NOT the same one as the Main.py
    plt.savefig(f"Indonesia_A{ACCURACIES[i]}.png", bbox_inches='tight', pad_inches=0)
    plt.show()


#-------------------------------------------------------------------------------

## PLOTTING WITH A FIXED CUTOFF AND SHOWING THE POINTS THAT ARE FURTHEST AWAY FROM EACH OTHER IN THE ORIGINAL AND NEW IMAGE
filtered = fourier.filter_coeffs(coeffs,5)
inverse = fourier.inverse_fourier_descriptors(filtered)
pointer_accuracy,index = met.calc_metric(inverse,image_1.points,mtype='Accuracy')
longest_distance = [(np.real(image_1.points[index[0][0]]),np.imag(image_1.points[index[0][0]])),(np.real(inverse[index[1][0]]),np.imag(inverse[index[1][0]]))] # the coordinates of the points that are furthest away from each other in the original and new image
print("pointer_accuracy: ", pointer_accuracy)

plt.plot(np.real(inverse),np.imag(inverse),label=f'Reconstructed Image')
plt.plot(np.real(image_1.points),np.imag(image_1.points),label=f'Original Image')
plt.plot(np.real(image_1.points[index[0][0]]),np.imag(image_1.points[index[0][0]]),'ro',label='Furthest point from new image')
plt.plot(np.real(inverse[index[1][0]]),np.imag(inverse[index[1][0]]),'go',label='Furthest point from original image')
plt.plot([longest_distance[0][0],longest_distance[1][0]],[longest_distance[0][1],longest_distance[1][1]],'k--',label='Longest distance between points') 

plt.title("Representation of Hausdorf distance between original and new image", fontsize=20)
plt.axis('off')
plt.legend(fontsize=17,title="Legend", title_fontsize=19)    
plt.show()



#--------------------------------------------------------------------------------------------

## PLOTTING THE ACCURACY OVER FREQUENCY CUTOFFS
#plotting the accuracy over the every frequency cutoff from 1 to N/2
frequencies = np.linspace(1,N_SAMPLES/2,N_SAMPLES//2)
for k in frequencies:
    
    k = int(k) #cast to int as the frequencies are floats and we need to use them as integers for the filter function
    filtered = fourier.filter_coeffs(coeffs,k)
    inverse = fourier.inverse_fourier_descriptors(filtered)
    pointer_accuracy,index = met.calc_metric(inverse,image_1.points,mtype='Accuracy')
    plt.scatter(k,pointer_accuracy,color='b')
plt.scatter(best_k,ACCURACIES,color='r',label='Best K for each accuracy threshold')

plt.title('Accuracy over frequency cutoffs')
plt.xlabel('Frequency cutoff')
plt.ylabel('Accuracy')
plt.show()

#---------------------------------------------------------------------------------------------












