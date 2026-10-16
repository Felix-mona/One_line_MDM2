#importing svg files and outputting coords
def import_svg(file_path):
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
