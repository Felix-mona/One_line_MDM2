import Metric as met
from SVG_Manipulation import Image
import Fourier_reconsctruction as fourier
import matplotlib.pyplot as plt

FILEPATH = r"C:\Users\felix\Uni\Year 2\MDM2\Primitive_line_project\Project_Files\One_line_MDM2\Example_images\building_skyline.svg"
N_SAMPLES = 200
ALPHA_BETA = [(0,1),
              (1,0),
              (0.5,0.5),
              (0.7,0.3),
              (0.3,0.7)]
FREQUENCY_BANDS = 5

image_1 = Image(FILEPATH,N_SAMPLES)
image_1.sample_paths() #should fill the object with the complex coords

fourier_descriptors,frequencies = fourier.fourier_coeffs(image_1.points)

lowest_loss_image = met.gather_lowest_metric(image_1.points,N_SAMPLES/2,FREQUENCY_BANDS,ALPHA_BETA)














