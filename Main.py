import Metric as met
from SVG_Manipulation import Image
import Fourier_reconsctruction as fourier
import matplotlib.pyplot as plt

"""
Test
"""
"""FILEPATH = "Example_images/smiley.svg"
N_SAMPLES = 200
ALPHA_BETA = [(0,1),
              (1,0),
              (0.5,0.5),
              (0.7,0.3),
              (0.3,0.7)]
FREQUENCY_BANDS = 5

image_1 = Image("Example_images/smiley.svg",N_SAMPLES)
image_1.sample_paths() #should fill the object with the complex coords

fourier_descriptors,frequencies = fourier.fourier_coeffs(image_1.points)

lowest_loss_image = met.gather_lowest_metric(image_1.points,N_SAMPLES/2,FREQUENCY_BANDS,ALPHA_BETA)

#plotting
original_x,original_y = [c.real for c in image_1.points],[c.imag for c in image_1.points]
shape1 = fourier.inverse_fourier_descriptors(fourier.filter_coeffs(fourier_descriptors,lowest_loss_image[0][0]))
shape2 = fourier.inverse_fourier_descriptors(fourier.filter_coeffs(fourier_descriptors,lowest_loss_image[1][0]))
shape3 = fourier.inverse_fourier_descriptors(fourier.filter_coeffs(fourier_descriptors,lowest_loss_image[2][0]))
s1_x,s1_y = [c.real for c in shape1],[c.imag for c in shape1]
s2_x,s2_y = [c.real for c in shape2],[c.imag for c in shape2]
s3_x,s3_y = [c.real for c in shape3],[c.imag for c in shape3]




plt.figure(figsize=(10,10))
plt.plot(original_x,original_y,color='red')
plt.plot(s1_x,s1_y,color='blue')
plt.plot(s2_x,s2_y,color='yellow')
plt.plot(s3_x,s3_y,color='black')

plt.gca().invert_yaxis()
plt.axis('equal')
plt.show()

"""


### using Claude plotting file, could be more helpful than any plots that we could produce
import Plotting as P


image = Image("Example_images/smiley.svg", 200); image.sample_paths()
pts = image.points

P.plot_spectrum(pts)                         # where is the energy?
P.plot_reconstructions(pts, [1, 3, 5, 10, 20, 50, 100])   # small multiples
res = P.sweep(pts, alpha_beta_pairs=[(1, 0), (0.5, 0.5), (0.3, 0.7)])
P.plot_metrics(res)                          # accuracy / smoothness / loss vs k
P.plot_best_overlay(pts, res["best"])        # best k for each alpha/beta
slider = P.interactive_explorer(pts)         # drag k and watch the curve










