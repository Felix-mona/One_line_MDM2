import numpy as np
import matplotlib.pyplot as plt
from svgpathtools import svg2paths
import svgwrite
from pathlib import Path

base_dir = Path.cwd()
npy_path = base_dir / "outputs" / "Indonesia" / "Indonesia_output_polyline.npy"
svg_path = base_dir / "outputs" / "Indonesia" / "Indonesia_output_polyline.svg"

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
# -----------