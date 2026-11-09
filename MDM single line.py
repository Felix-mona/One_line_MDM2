import numpy as np
import matplotlib.pyplot as plt
from svgpathtools import svg2paths

from pathlib import Path

# Always points to the folder where your script/terminal is operating from
base_dir = Path.cwd()

npy_path = base_dir / "outputs" / "Shapes" / "Shapes_output_polyline.npy"
svg_path = base_dir / "outputs" / "Shapes" / "Shapes_output_polyline.svg"

paths, attributes = svg2paths(svg_path)


complex_coords = []

for path in paths:
    

    for i in range(1001):
        t = i / 1000        
        point = path.point(t)    #  SVG for the exact (X, Y) coordinate at that specific fraction
        complex_coords.append(point)  # Save that coordinate to list


complex_coords = np.array(complex_coords)


complex_coords = np.concatenate([complex_coords, complex_coords[::-1]])

fourier_descriptors = np.fft.fft(complex_coords) #converting to fourier



fourier_filtered = fourier_descriptors.copy()


#keep_low_frequencies = 0.2
#n = len(fourier_filtered)
#keep_count = int(n * keep_low_frequencies)

keep_count = 200

fourier_filtered[keep_count:-keep_count] = 0


new_complex_coords = np.fft.ifft(fourier_filtered) #reverse fourier


orig_x = [c.real for c in complex_coords]
orig_y = [c.imag for c in complex_coords]

#complex back into x and y
smooth_x = [c.real for c in new_complex_coords]
smooth_y = [c.imag for c in new_complex_coords]

# Plotting both original and new
plt.figure(figsize=(10, 10))
plt.plot(orig_x, orig_y, color='red',  linewidth=4)
plt.plot(smooth_x, smooth_y , color='blue', linewidth=2)

plt.gca().invert_yaxis()  # Flip it as keeps being upside down
plt.axis('equal') #no stretch

plt.show()