from os import path

import numpy as np
from svgpathtools import svg2paths 



#importing svg files
def import_svg(file_path):

    paths, attributes = svg2paths(file_path)

    return paths, attributes

# taking svg paths and sampling k along the path to get k different vector points along the path with equal spacing
def sample_path(paths, k):

    L = paths[0].length()

    distances = np.linspace(0, L, k)

    points = []

    for s in distances:

        t = paths[0].ilength(s)

        z = paths[0].point(t)

        points.append((z.real, z.imag))

    return np.array(points)
