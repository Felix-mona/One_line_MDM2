import numpy as np
import matplotlib.pyplot as plt
from svgpathtools import svg2paths
import svgwrite
from pathlib import Path

base_dir = Path.cwd()

figure_folder = base_dir / "outputs" / "fourier_figures"
figure_folder.mkdir(parents=True, exist_ok=True) 


npy_path = base_dir / "outputs" / "HedgehogMDM" / "HedgehogMDM_output_polyline.npy"
svg_path = base_dir / "outputs" / "HedgehogMDM" / "HedgehogMDM_output_polyline.svg"

# Ensure output directory exists
svg_path.parent.mkdir(parents=True, exist_ok=True)

# Load numpy array and convert to points list
npy_coords = np.load(npy_path)
points = [
    (float(c.real), float(c.imag)) if isinstance(c, complex) else (float(c[0]), float(c[1]))
    for c in npy_coords
]

# Write out the .svg file
dwg = svgwrite.Drawing(str(svg_path), size=( "2000px", "2000px" ))
dwg.add(dwg.polyline(points=points, stroke="black", fill="none", stroke_width="1"))
dwg.save()
# -----------------------------------------------


paths, attributes = svg2paths(svg_path)

number_of_frequencies_kept_2 = 50


complex_coords = []

for path in paths:
    

    for i in range(10000):
        t = i / 10000        
        point = path.point(t)    #  SVG for the exact (X, Y) coordinate at that specific fraction
        complex_coords.append(point)  # Save that coordinate to list


complex_coords = np.array(complex_coords)




fourier_descriptors = np.fft.fft(complex_coords) #converting to fourier



fourier_filtered = fourier_descriptors.copy()


#keep_low_frequencies = 0.2
#n = len(fourier_filtered)
#keep_count = int(n * keep_low_frequencies)


number_of_frequencies_kept = number_of_frequencies_kept_2 // 2
#fourier_filtered[number_of_frequencies_kept:-number_of_frequencies_kept] = 0
fourier_filtered[number_of_frequencies_kept:] = 0

new_complex_coords = np.fft.ifft(fourier_filtered) #reverse fourier


orig_x = [c.real for c in complex_coords]
orig_y = [c.imag for c in complex_coords]

#complex back into x and y
smooth_x = [c.real for c in new_complex_coords]
smooth_y = [c.imag for c in new_complex_coords]

# Plotting both original and new
plt.figure(figsize=(10, 10))
#plt.plot(orig_x, orig_y, color='red',  linewidth=4)
plt.plot(smooth_x, smooth_y , color='blue', linewidth=2)

#plt.gca().invert_yaxis()  # Flip it as keeps being upside down
plt.axis('equal') #no stretch


plt.savefig(figure_folder / f"HedgeHog2_{number_of_frequencies_kept_2}.png", dpi=200, bbox_inches="tight")



plt.show()