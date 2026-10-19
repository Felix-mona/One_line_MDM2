#importing svg files and outputting coords
def import_svg(file_path):
<<<<<<< Updated upstream
    # not done yet, just understanding github
    coords = []
    with open(file_path, 'r') as file:
        for line in file:
            if '<path' in line:
                # Extract the 'd' attribute from the path element
                start_index = line.find('d="') + 3
                end_index = line.find('"', start_index)
                path_data = line[start_index:end_index]
                
                # Split the path data into individual commands and coordinates
                commands = path_data.split()
                for command in commands:
                    if command[0].isalpha():  # Check if it's a command (e.g., M, L, C)
                        continue
                    else:
                        # Convert coordinate strings to float and append to coords list
                        coords.append(tuple(map(float, command.split(','))))
    return coords
=======

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

paths, attributes = import_svg('example_images/gapped_curve.svg')

points = sample_paths(paths, 20)
plt.scatter(points[:, 0], points[:, 1])
plt.show()
>>>>>>> Stashed changes
