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

Plots for the survey:
- plots of each curve at same accuracy and smoothness thresholds, labeled
- check for the thresholds, they might be different for each image.\\
- plot the accuracy and smoothness over frequencies for every image\\
- hausdorf distance thats been taken 
- graph of the different thresholds cutting through the smoothness and accuracy.\\
 

0.12
 Indonesia, min0.01, max 0.106, 5 spacing 

Calculated percentiles for each image:
 - Bridge - 
"""
FILEPATH = r"C:\Users\felix\Uni\Year 2\MDM2\Primitive_line_project\Project_Files\One_line_MDM2\Testing_images\flower_5_petal.svg"
N_SAMPLES = 1000
IMG = "5 petal flower"


image_1 = Image(str(FILEPATH),N_SAMPLES)
image_1.sample_paths() # fills the object with the complex coords
coeffs, freqs = fourier.forward_fourier(image_1.points) #gathers complex coefficients and frequencies from the original image

#Plot to explain the hausdorf distance, one when it is a good metric and one where it is not.
## PLOTTING WITH A FIXED CUTOFF AND SHOWING THE POINTS THAT ARE FURTHEST AWAY FROM EACH OTHER IN THE ORIGINAL AND NEW IMAGE

def plot_hausdorff(original, coeffs, k, title=None, ax=None):
    filtered = fourier.filter_coeffs(coeffs, k)
    new = fourier.inverse_fourier_descriptors(filtered)

    accuracy, (i_orig, i_new, direction) = met.calc_metric(new, original, mtype='Accuracy')
    p_orig, p_new = original[i_orig], new[i_new]          # complex numbers

    # which curve is the "problem" side depends on the direction
    if direction == 'original_to_reconstructed':
        lab_o, lab_n = 'Original point furthest from new image', 'Nearest point on new image'
    else:
        lab_o, lab_n = 'Nearest point on original image', 'New point furthest from original image'

    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(original.real, original.imag, color='0.65', lw=3, label='Original image')
    ax.plot(new.real, new.imag, color='tab:blue', lw=1.5, label=f'Reconstructed image (k={k})')
    ax.plot([p_orig.real, p_new.real], [p_orig.imag, p_new.imag],
            'k--', lw=2, label=f'Hausdorff distance = {accuracy:.3f}')
    ax.plot(p_orig.real, p_orig.imag, 'ro', ms=10, label=lab_o)
    ax.plot(p_new.real, p_new.imag, 'go', ms=10, label=lab_n)

    ax.invert_yaxis()            # SVG y axis points down
    ax.set_aspect('equal')       # essential, or the distance looks distorted
    ax.axis('off')
    ax.set_title(title or f'Hausdorff distance, k={k}', fontsize=20)
    ax.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, 0.0), frameon=False)
    if show:
        plt.show()
    return accuracy, direction

fig, axes = plt.subplots(1, 2, figsize=(15, 9))
fig.subplots_adjust(bottom=0.3, wspace=0.15)      # bottom = legend room, wspace = gap between panels

plot_hausdorff(image_1.points, coeffs, 4, "Poor fit", ax=axes[0])
plot_hausdorff(image_1.points, coeffs, 12, "Accurate fit", ax=axes[1])
plt.show()


def plot_Smoothness(original=None, coeffs=None, k=None, title=None, ax=None):

    # -------------------------------------------------
    # 1. Generate a simple S-shaped curve at 45 degrees
    # -------------------------------------------------

    t = np.linspace(-2, 2, 200)

    x = t
    y = 0.35 * np.cos(6 * t) + t

    # Rotate the curve by 45 degrees
    theta = np.radians(45)

    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)

    curve = x_rot + 1j * y_rot

    # -------------------------------------------------
    # 2. Select three consecutive points
    # -------------------------------------------------

    # Select points around the centre of the curve
    i = 100

    p1 = curve[i - 5]
    p2 = curve[i]
    p3 = curve[i + 5]

    # Convert to Cartesian coordinates
    points = np.array([
        [p1.real, p1.imag],
        [p2.real, p2.imag],
        [p3.real, p3.imag]
    ])

    # -------------------------------------------------
    # 3. Calculate the three side lengths
    # -------------------------------------------------

    a = np.linalg.norm(points[0] - points[2])
    b = np.linalg.norm(points[1] - points[2])
    c = np.linalg.norm(points[1] - points[0])

    # -------------------------------------------------
    # 4. Calculate the angle at the middle point
    #    Using the same cosine rule as metric.py
    # -------------------------------------------------

    cos_angle = (b**2 + c**2 - a**2) / (2 * b * c)

    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle = np.arccos(cos_angle)

    angle_degrees = np.degrees(angle)

    smoothness_contribution = 1 - angle / np.pi

    # -------------------------------------------------
    # 5. Set up plot
    # -------------------------------------------------

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Plot original S-shaped curve
    ax.plot(
        x_rot,
        y_rot,
        color='steelblue',
        linewidth=2,
        label='S-shaped curve'
    )

    # -------------------------------------------------
    # 6. Plot the three selected points
    # -------------------------------------------------

    ax.scatter(
        points[:, 0],
        points[:, 1],
        color=['red', 'green', 'red'],
        s=90,
        zorder=5
    )

    ax.annotate('Previous point',
                points[0],
                xytext=(-35, -20),
                textcoords='offset points')

    ax.annotate('Current point',
                points[1],
                xytext=(10, 15),
                textcoords='offset points')

    ax.annotate('Next point',
                points[2],
                xytext=(10, -20),
                textcoords='offset points')

    # -------------------------------------------------
    # 7. Draw the three sides of the triangle
    # -------------------------------------------------

    ax.plot(
        [points[0, 0], points[1, 0]],
        [points[0, 1], points[1, 1]],
        'r--',
        linewidth=1.8,
        label='Previous to current'
    )

    ax.plot(
        [points[1, 0], points[2, 0]],
        [points[1, 1], points[2, 1]],
        'g--',
        linewidth=1.8,
        label='Current to next'
    )

    ax.plot(
        [points[0, 0], points[2, 0]],
        [points[0, 1], points[2, 1]],
        'k:',
        linewidth=1.8,
        label='Previous to next'
    )

    # -------------------------------------------------
    # 8. Draw the angle at the middle point
    # -------------------------------------------------

    v1 = points[0] - points[1]
    v2 = points[2] - points[1]

    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])

    # Find the smaller angular arc
    difference = (angle2 - angle1 + np.pi) % (2 * np.pi) - np.pi

    arc_angles = np.linspace(
        angle1,
        angle1 + difference,
        100
    )

    radius = 0.25 * min(a, c)

    arc_x = points[1, 0] + radius * np.cos(arc_angles)
    arc_y = points[1, 1] + radius * np.sin(arc_angles)

    ax.plot(
        arc_x,
        arc_y,
        color='purple',
        linewidth=2.5,
        label='Measured angle'
    )

    # Label angle
    mid_angle = angle1 + difference / 2

    label_x = points[1, 0] + 1.6 * radius * np.cos(mid_angle)
    label_y = points[1, 1] + 1.6 * radius * np.sin(mid_angle)

    ax.text(
        label_x,
        label_y,
        f'{angle_degrees:.1f}°',
        color='purple',
        fontsize=13,
        fontweight='bold'
    )

    # -------------------------------------------------
    # 9. Display calculated smoothness contribution
    # -------------------------------------------------

    ax.text(
        0.03,
        0.97,
        f'Angle: {angle_degrees:.2f}°\n'
        f'Smoothness contribution: {smoothness_contribution:.4f}',
        transform=ax.transAxes,
        verticalalignment='top',
        fontsize=11,
        bbox=dict(
            facecolor='white',
            edgecolor='grey',
            alpha=0.9
        )
    )

    # -------------------------------------------------
    # 10. Formatting
    # -------------------------------------------------

    ax.set_title(
        title or 'Visualisation of the Smoothness Metric'
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax.axis('equal')
    ax.grid(alpha=0.2)
    ax.legend()

    if ax is None:
        plt.show()

    return angle_degrees, smoothness_contribution


"""plot_Smoothness(original=image_1.points,coeffs=coeffs, k=10,title='Smoothness visualisation',ax=None)

# Create the figure
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the S-shaped curve and smoothness calculation
angle, smoothness = plot_Smoothness(original=image_1.points,coeffs=coeffs, k=10,title='Smoothness visualisation',ax=ax)

# Display results
print(f"Calculated angle: {angle:.2f} degrees")
print(f"Smoothness contribution: {smoothness:.4f}")

plt.show()"""















#--------------------------------Plotting the different thesholds on the graph and then also making an image per smoothness and accuracy threshold --------------
#Used in the survey mainly
"""
## Plotting the smoothness over k and the accuracy over k
best_k = [] # initialised to be filled with the best frequency cutoffs for each accuracy threshold
best_k_indexes = [] # holds the coords where the best k is found, for plotting the shortest distance between the two curves


frequencies = np.linspace(1,N_SAMPLES/2,N_SAMPLES//2)
smoothness = []
accuracy = []
for k in frequencies:
    k = int(k) #cast to int as the frequencies are floats and we need to use them as integers for the filter function
    filtered = fourier.filter_coeffs(coeffs,k)
    inverse = fourier.inverse_fourier_descriptors(filtered)
    pointer_accuracy,index = met.calc_metric(inverse,image_1.points,mtype='Accuracy')
    pointer_smoothness = met.calc_metric(inverse,image_1.points,mtype='Smoothness')
    smoothness.append(pointer_smoothness)
    accuracy.append(pointer_accuracy)


smooth_5 = np.percentile(smoothness, 5)
smooth_95 = np.percentile(smoothness, 95)

THRESHOLDS = np.linspace(smooth_5, smooth_95, 5)

fig, ax = plt.subplots()

# Plot smoothness and accuracy
ax.plot(frequencies, smoothness, color='b', label='Smoothness')
ax.plot(frequencies, accuracy, color='g', label='Accuracy')

# Plot intermediate thresholds
for i in range(1, 4):
    ax.axhline(y=THRESHOLDS[i], color='orange', linestyle='--')

# Plot percentile boundaries
ax.axhline(y=smooth_5, color='r', linestyle='--', label='5th percentile')
ax.axhline(y=smooth_95, color='r', linestyle='--', label='95th percentile')

# Add threshold labels to the right of the graph
for i, threshold in enumerate(THRESHOLDS):
    label = f'T{i+1}: {threshold:.3f}'

    if i == 0:
        label = f'5th percentile: {threshold:.3f}'
    elif i == 4:
        label = f'95th percentile: {threshold:.3f}'

    ax.text(
        1.02,
        threshold,
        label,
        transform=ax.get_yaxis_transform(),
        va='center',
        fontsize=9
    )

# Formatting
ax.set_title('Accuracy over frequency cutoffs')
ax.set_xlabel('Frequency cutoff')
ax.set_ylabel('Accuracy/Smoothness')
ax.legend()

# Make room for labels on the right
plt.subplots_adjust(right=0.75)
plt.savefig(f'{IMG}_percentiles.png', bbox_inches='tight', pad_inches=0)
plt.show()

THRESHOLDS = np.round(THRESHOLDS,3)

# basic loop that increases the frequency cutoff by 1 until the accuracy threshold is reached, or the maximum frequency cutoff is reached. The best frequency cutoff for each accuracy threshold is stored in best_k
for thresh in THRESHOLDS:
    max_pointer = int(N_SAMPLES//2)
    min_pointer = 1
    accuracy_pointer = 1 #frequency cutoff for accuracies
    smoothness_pointer = N_SAMPLES//2 # frquency cut off for smoothness
    temp_accuracy = 10
    temp_smoothness = 10

    while temp_accuracy > thresh:
        filtered = fourier.filter_coeffs(coeffs,accuracy_pointer) # filters the coefficients based on the current frequency cutoff
        inverse = fourier.inverse_fourier_descriptors(filtered)
        temp_accuracy,index = met.calc_metric(inverse,image_1.points,mtype='Accuracy')

        if temp_accuracy < thresh: # case where the accuracy has been reached
            break
        elif accuracy_pointer >= max_pointer: # case where the frequency cutoff has reached the maximum and the accuracy has not been reached
            print("max frequency cutoff reached, accuracy not reached")
            accuracy_pointer = -1 # set the accuracy_pointer to -1 to indicate that the accuracy was not reached
            break
        else:
            accuracy_pointer += 1 # increment the accuracy_pointer to increase the frequency cutoff and get a better accuracy

    best_k.append(accuracy_pointer)
    best_k_indexes.append(index)
    print("Working...accuracies")

    while temp_smoothness > thresh:
            filtered = fourier.filter_coeffs(coeffs,smoothness_pointer)
            inverse = fourier.inverse_fourier_descriptors(filtered)
            temp_smoothness = met.calc_metric(inverse,image_1.points,mtype='Smoothness')
    
            if temp_smoothness < thresh: # case where the smoothness has been reached
                break
            elif smoothness_pointer <= min_pointer: # case where the frequency cutoff has reached the minimum and the smoothness has not been reached
                print("min frequency cutoff reached, smoothness not reached")
                smoothness_pointer = -1 # set the smoothness_pointer to -1 to indicate that the smoothness was not reached
                break
            else:
                smoothness_pointer -= 1 # increment the smoothness_pointer to increase the frequency cutoff and get a better accuracy
    
    best_k.append(smoothness_pointer)
    print("Working...smoothness")



#plotting loop that will also write the images to a file, use no padding and large font size otherwise hard to see on overleaf
# plots over the 5 different accuracy thresholds, change range to 1 and comment out the new image related code to get the original plotted out
# comment out lines that you do not need
for i in range(10):
    complex_coords = fourier.inverse_fourier_descriptors(fourier.filter_coeffs(coeffs, best_k[i]))
    #longest_distance = [(np.real(image_1.points[index[0][0]]),np.imag(image_1.points[index[0][0]])),(np.real(inverse[index[1][0]]),np.imag(inverse[index[1][0]]))] 
    x,y = np.real(complex_coords), np.imag(complex_coords)

    # Plotting the new image over the original image
    plt.plot(x,y,label=f'{best_k[i]} K cutoff')
    #plt.plot(np.real(image_1.points),np.imag(image_1.points),label=f'Original Image')

    
    plt.title(f'{IMG}', fontsize=20)
    plt.axis('off')
    #plt.legend()

    #saves the plot created within the current directory, NOT the same one as the Main.py
    if i%2 == 0:
        plt.savefig(f'{IMG}_A{THRESHOLDS[i//2]}.png', bbox_inches='tight', pad_inches=0)
        print("Saving image with accuracy threshold: ", THRESHOLDS[i//2])
    else:
        plt.savefig(f'{IMG}_S{THRESHOLDS[i//2]}.png', bbox_inches='tight', pad_inches=0)
        print("Saving image with smoothness threshold: ", THRESHOLDS[i//2])

    plt.show()
"""



































#-------------------------------------------------------------------------------
"""
## PLOTTING WITH A FIXED CUTOFF AND SHOWING THE POINTS THAT ARE FURTHEST AWAY FROM EACH OTHER IN THE ORIGINAL AND NEW IMAGE
filtered = fourier.filter_coeffs(coeffs,30)
inverse = fourier.inverse_fourier_descriptors(filtered)
pointer_accuracy,index = met.calc_metric(inverse,image_1.points,mtype='Accuracy')
longest_distance = [(np.real(image_1.points[index[0][0]]),np.imag(image_1.points[index[0][0]])),(np.real(inverse[index[1][0]]),np.imag(inverse[index[1][0]]))] # the coordinates of the points that are furthest away from each other in the original and new image
print("pointer_accuracy: ", pointer_accuracy)

plt.plot(np.real(inverse),np.imag(inverse),label=f'Reconstructed Image')
#plt.plot(np.real(image_1.points),np.imag(image_1.points),label=f'Original Image')
#plt.plot(np.real(image_1.points[index[0][0]]),np.imag(image_1.points[index[0][0]]),'ro',label='Furthest point from new image')
#plt.plot(np.real(inverse[index[1][0]]),np.imag(inverse[index[1][0]]),'go',label='Furthest point from original image')
#plt.plot([longest_distance[0][0],longest_distance[1][0]],[longest_distance[0][1],longest_distance[1][1]],'k--',label='Longest distance between points') 

plt.title("Representation of Hausdorf distance between original and new image", fontsize=20)
plt.axis('off')
plt.legend(fontsize=17,title="Legend", title_fontsize=19)    
plt.show()

"""
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
"""
#---------------------------------------------------------------------------------------------
""""


## Plotting the smoothness over k and the accuracy over k

frequencies = np.linspace(1,N_SAMPLES/2,N_SAMPLES//2)
smoothness = []
accuracy = []
for k in frequencies:
    
    k = int(k) #cast to int as the frequencies are floats and we need to use them as integers for the filter function
    filtered = fourier.filter_coeffs(coeffs,k)
    inverse = fourier.inverse_fourier_descriptors(filtered)
    pointer_accuracy,index = met.calc_metric(inverse,image_1.points,mtype='Accuracy')
    pointer_smoothness = met.calc_metric(inverse,image_1.points,mtype='Smoothness')
    smoothness.append(pointer_smoothness)
    accuracy.append(pointer_accuracy)

#plt.scatter(best_k,ACCURACIES,color='r',label='Best K for each accuracy threshold')

smooth_5 = np.percentile(smoothness, 5)
smooth_95 = np.percentile(smoothness, 95)

plt.axhline(y=smooth_5, color='r', linestyle='--', label='5th percentile')
plt.axhline(y=smooth_95, color='r', linestyle='--', label='95th percentile')


plt.plot(frequencies,smoothness,color='b',label='Smoothness')
plt.plot(frequencies,accuracy,color='g',label='Accuracy')
plt.title('Accuracy over frequency cutoffs')
plt.xlabel('Frequency cutoff')
plt.ylabel('Accuracy/Smoothness')
plt.legend()
plt.show()

print("5th percentile smoothness: ", smooth_5)
print("95th percentile smoothness: ", smooth_95)

"""


"""
# basic loop that increases the frequency cutoff by 1 until the accuracy threshold is reached, or the maximum frequency cutoff is reached. The best frequency cutoff for each accuracy threshold is stored in best_k
for smoothness_thresh in SMOOTHNESS:
    min_pointer = 1
    pointer_k = int(N_SAMPLES/2) #frequency cutoff to start with, will be changed in the while loop
    pointer_smoothness = 10

    while pointer_smoothness > smoothness_thresh:
        filtered = fourier.filter_coeffs(coeffs,pointer_k)
        inverse = fourier.inverse_fourier_descriptors(filtered)
        pointer_smoothness = met.calc_metric(inverse,image_1.points,mtype='Smoothness')

        if pointer_smoothness < smoothness_thresh: # case where the smoothness has been reached
            break
        elif pointer_k <= min_pointer: # case where the frequency cutoff has reached the minimum and the smoothness has not been reached
            print("min frequency cutoff reached, smoothness not reached")
            pointer_k = -1 # set the pointer_k to -1 to indicate that the smoothness was not reached
            break
        else:
            pointer_k -= 1 # increment the pointer_k to increase the frequency cutoff and get a better accuracy

    best_k.append(pointer_k)
    print("Working...")

#plotting loop that will also write the images to a file, use no padding and large font size otherwise hard to see on overleaf
# plots over the 5 different smoothness thresholds, change range to 1 and comment out the new image related code to get the original plotted out
# comment out lines that you do not need
for i in range(5):
    complex_coords = fourier.inverse_fourier_descriptors(fourier.filter_coeffs(coeffs, best_k[i]))
    #longest_distance = [(np.real(image_1.points[index[0][0]]),np.imag(image_1.points[index[0][0]])),(np.real(inverse[index[1][0]]),np.imag(inverse[index[1][0]]))] 
    x,y = np.real(complex_coords), np.imag(complex_coords)

    # Plotting the new image over the original image
    plt.plot(x,y,label=f'{best_k[i]} K cutoff')
    #plt.plot(np.real(image_1.points),np.imag(image_1.points),label=f'Original Image')

    # plots the points where the hausdorf distance was taken from
    #plt.plot(np.real(image_1.points[index[0][0]]),np.imag(image_1.points[index[0][0]]),'ro',label='Furthest point from new image')
    #plt.plot(np.real(inverse[index[1][0]]),np.imag(inverse[index[1][0]]),'go',label='Furthest point from original image')

    #plots the line between the two points for the hausdorf distance
    #plt.plot([longest_distance[0][0],longest_distance[1][0]],[longest_distance[0][1],longest_distance[1][1]],'k--',label='Longest distance between points') 

    plt.title(f'Indonesia smoothness {SMOOTHNESS[i]}', fontsize=20)
    plt.axis('off')
    plt.legend()

    #saves the plot created within the current directory, NOT the same one as the Main.py
    #plt.savefig(f"Indonesia_S{SMOOTHNESS[i]}.png", bbox_inches='tight', pad_inches=0)
    plt.show()


"""



