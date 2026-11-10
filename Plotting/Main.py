import Metric as met
from SVG_Manipulation import Image
import Fourier_reconsctruction as fourier
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import svgwrite


"""
Test

AIMS:
To plot 5 images, ranging in differing accuracy thresholds. (with each having a selected frequency)

Accuracy - the hausdorf distance, between x and y (0,1)
Fixed decided accuracies - 0.1,0.2,0.3,0.4,0.5

for each accuracy need to find the minimum frequency cutoff that reaches it before going over the threshold




"""


FILEPATH = r"outputs\bridge\bridge_output_polyline.svg"
N_SAMPLES = 1000
ACCURACIES = [(0.002),
              (0.005),
              (0.05),
              (0.1),
              (0.2)]

image_1 = Image(str(FILEPATH),N_SAMPLES)
image_1.sample_paths() #should fill the object with the complex coords

coeffs, freqs = fourier.forward_fourier(image_1.points)
best_k = []
best_k_indexes = []


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


for i in range(5):
    complex_coords = fourier.inverse_fourier_descriptors(fourier.filter_coeffs(coeffs, best_k[i]))
    longest_distance = [(np.real(image_1.points[index[0][0]]),np.imag(image_1.points[index[0][0]])),(np.real(inverse[index[1][0]]),np.imag(inverse[index[1][0]]))] 
    x,y = np.real(complex_coords), np.imag(complex_coords)

    # Plotting the new image over the original image
    plt.plot(x,y,label=f'{best_k[i]} K cutoff')
    #plt.plot(np.real(image_1.points),np.imag(image_1.points),label=f'Original Image')

    # plots the points where the hausdorf distance was taken from
    #plt.plot(np.real(image_1.points[index[0][0]]),np.imag(image_1.points[index[0][0]]),'ro',label='Furthest point from new image')
    #plt.plot(np.real(inverse[index[1][0]]),np.imag(inverse[index[1][0]]),'go',label='Furthest point from original image')

    #plots the line between the two points for the hausdorf distance
    #plt.plot([longest_distance[0][0],longest_distance[1][0]],[longest_distance[0][1],longest_distance[1][1]],'k--',label='Longest distance between points') 

    plt.title(f'Accuracy threshold: {ACCURACIES[i]}')
    plt.legend()
    plt.show()


"""
#-------------------------------------------------------------------------------

## PLOTTING WITH A FIXED CUTOFF AND SHOWING THE POINTS THAT ARE FURTHEST AWAY FROM EACH OTHER IN THE ORIGINAL AND NEW IMAGE
filtered = fourier.filter_coeffs(coeffs,CUTOFF)
inverse = fourier.inverse_fourier_descriptors(filtered)
pointer_accuracy,index = met.calc_metric(inverse,image_1.points,mtype='Accuracy')
longest_distance = [(np.real(image_1.points[index[0][0]]),np.imag(image_1.points[index[0][0]])),(np.real(inverse[index[1][0]]),np.imag(inverse[index[1][0]]))] # the coordinates of the points that are furthest away from each other in the original and new image
print("pointer_accuracy: ", pointer_accuracy)

plt.plot(np.real(inverse),np.imag(inverse),label=f'K cutoff: {5}, Accuracy: {pointer_accuracy}')
plt.plot(np.real(image_1.points),np.imag(image_1.points),label=f'Original Image')
plt.plot(np.real(image_1.points[index[0][0]]),np.imag(image_1.points[index[0][0]]),'ro',label='Furthest point from new image')
plt.plot(np.real(inverse[index[1][0]]),np.imag(inverse[index[1][0]]),'go',label='Furthest point from original image')
plt.plot([longest_distance[0][0],longest_distance[1][0]],[longest_distance[0][1],longest_distance[1][1]],'k--',label='Longest distance between points') 
plt.legend()    
plt.show()

"""

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
"""
per image atm, going to change the image by hand
image_coeffs = fourier.forward()

For each accuracy:
    best_k = 0 # ready to be changed to the highest k cutoff value
    max_pointer = N/2
    lower_pointer = 1
    pointer = N/4

    While pointer_accuracy > accuracy:

        filtered = fourier.filter(pointer, image_coeffs)
        inverse = fourier.inv(filtered)
        metric = calc_metric()

        if metric < accuracy:
            pointer = (pointer + lower_pointer) / 2
            max_pointer = pointer
        
        else:
            pointer = (pointer + max_pointer) / 2
            lower_pointer = pointer
    
    best_k.append(pointer)

plot each new drawing with each best_k cutoff
"""














