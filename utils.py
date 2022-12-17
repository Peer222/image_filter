import re
from typing import List, Dict
from matplotlib import colors
from pathlib import Path
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
import random

import inspect
import os
from datetime import datetime

class DropoutType():
    NORMAL = 0
    BRIGHTNESS = 1

# parses a color into RGBA-255 format
# TODO hex colors
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

# base_color None or black results in full random colors -> otherwise random spread of colors around base color 
def get_random_color(base_color=None, spread:float=1.0) -> List[int]:
    if base_color: base_color = parse_color_format(base_color)
    if not base_color or np.sum(base_color[:3]) == 0:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return [r, g, b, 255]

    spread = int(spread * 256)

    get_random_value = lambda x : random.randint(max(0, x - spread), min(255, x + spread))#max(0, min(random.randint(x - color_spread, x + color_spread), 255))

    r = get_random_value(base_color[0])
    g = get_random_value(base_color[1])
    b = get_random_value(base_color[2])
    return [r, g, b, 255]

def downsample(img:Image.Image, size:int) -> Image.Image:
    ratio = min(size/img.size[0], size/img.size[1])
    new_size = (int(img.size[0] * ratio),  int(img.size[1] * ratio))
    img.thumbnail(new_size)#, resample=Image.Resampling.LANCZOS)
    return img

def get_filepaths(folder, img_path):
    if img_path and type(img_path) == str: img_path = Path(img_path)
    if type(folder) == str: folder = Path(folder)
    if folder and not folder.is_dir(): folder.mkdir(parents=True, exist_ok=True)
    return folder, img_path

# adds options to saved image metadata
def save_image(result_folder:Path, img:np.array or Image.Image, img_name:str or Path, options:Dict) -> Image.Image:
    if type(img) != Image.Image: img = Image.fromarray(img)

    caller = inspect.stack()[1].function # gets name of caller function
    if not img_name: img_name = caller

    if type(img_name) == type(Path('.')): img_name = img_name.stem

    if not result_folder: return img

    # create image metadata
    metadata = PngInfo()
    metadata.add_text('reference', 'https://github.com/Peer222/image_filter')
    metadata.add_text('timestamp', datetime.now().strftime("%Y-%m-%d_%H-%M"))
    metadata.add_text('filter', caller)
    for key in options.keys():
        if key in ['result_folder', 'img_path', 'img']: continue
        metadata.add_text(key, str(options[key]))

    filename = f'{img_name}_result'
    suffix = '.png'
    if os.path.isfile(result_folder / f'{filename}{suffix}'):
        counter = 2
        while os.path.isfile(result_folder / f'{filename}_{counter}{suffix}'):
            counter += 1
        filename += f'_{counter}'
    img.save(result_folder / (filename + suffix), pnginfo=metadata)

    return img

def get_metadata(img_path:Path or str=None, img:Image.Image=None) -> Dict:
    if img_path: img = Image.open(img_path)
    data = img.text#info['meta_to_read']
    print('\nFilename: ', img.filename)
    print('\nMetadata: ')
    for key in data.keys():
        print(f'    {key}: ', data[key])
    print('')
    return data

if __name__ == '__main__':
    pass