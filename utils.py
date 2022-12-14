import re
from typing import List
from matplotlib import colors
from pathlib import Path

# parses a color into RGBA-255 format
def parse_color_format(color) -> List[int]:
    if type(color) == int or type(color) == float or type(color) == str: color = [color]
    color = list(color)

    if color[0] == 'original': return None

    if type(color[0]) == str and not re.findall('\d', color[0]): 
        color = list(colors.to_rgba(color[0]))

    for i in range(len(color)):
        c = color[i]
        if type(c) == str and '.' in c: color[i] = float(c)
        elif type(c) == str: color[i] = int(c)

    if len(color) == 1: color.extend([color[0], color[0]])
    color = [int(c * 255) if type(c) == float else c for c in color]

    if len(color) == 3: color.append(255)
    if len(color) != 4 or any(color) < 0 or any(color) > 255: raise Exception(f'parsed color: {color} -> non-matching color format')
    return color

def get_filepaths(folder, img_path):
    if img_path and type(img_path) == str: img_path = Path(img_path)
    if type(folder) == str: folder = Path(folder)
    if folder and not folder.is_dir(): folder.mkdir(parents=True, exist_ok=True)
    return folder, img_path