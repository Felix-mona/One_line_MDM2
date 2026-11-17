from os import path
import numpy as np
from svgpathtools import svg2paths 

# create an image object through a file path and then the object will have its paths and attributes as attributes
class Image:
    """
    image class:
    Within the __init__ paths and attributes are extracted from the file location and stored as attributes within an image object
    To get the list of the comple coords, must use image.sample_paths()
    """
    def __init__(self,file_path,delta_s):
        """
        Instantiate func
        INPUT:
         - File_path, the location of the svg turning into a set of points
         - delta_s, the amount of points to split the curve into
        METHODS:
         - image.sample_paths(), Creates the list of points along the SVG file
        
        """
        self.paths, self.attributes = svg2paths(file_path)
        self.k = delta_s
        self.points = np.zeros(delta_s, dtype=complex)
        

    
    def sample_paths(self):
        """
        Call to instantiate a list of k amount of complex coords covering the whole of the svg image, to be used in further processing
        Returns the list of complex coords for the given SVG file
        """

        lengths = np.array([p.length() for p in self.paths])
        # done for if the image is diconnected and has multiple separate paths
        cumulative = np.cumsum(lengths)
        total_length = cumulative[-1]
        distances = np.linspace(0, total_length, self.k)

        for i,d in enumerate(distances):
            path_idx = np.searchsorted(cumulative, d)
            previous = 0 if path_idx == 0 else cumulative[path_idx-1]
            local_s = d - previous
            
            path = self.paths[path_idx]
            t = path.ilength(local_s)
            self.points[i] = path.point(t)

        return self.points
