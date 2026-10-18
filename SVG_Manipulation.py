from os import path

import numpy as np
from svgpathtools import svg2paths 
import matplotlib.pyplot as plt


#importing svg files
def import_svg(file_path):

    paths, attributes = svg2paths(file_path)

    return paths, attributes

# taking svg paths and sampling k along the path to get k different vector points along the path with equal spacing
def sample_paths(paths, k):

    lengths = np.array([p.length() for p in paths])
    # done for if the image is diconnected and has multiple separate paths

    cumulative = np.cumsum(lengths)

    total_length = cumulative[-1]

    distances = np.linspace(0, total_length, k)

    points = []

    for d in distances:

        path_idx = np.searchsorted(cumulative, d)

        previous = 0 if path_idx == 0 else cumulative[path_idx-1]

        local_s = d - previous

        path = paths[path_idx]

        t = path.ilength(local_s)

        z = path.point(t)

        points.append([z.real, z.imag])

    return np.array(points)

paths, attributes = import_svg('example_images/circle.svg')

points = sample_paths(paths, 100)
plt.scatter(points[:, 0], points[:, 1])
plt.show()