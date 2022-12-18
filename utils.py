import re
from typing import List, Dict
from matplotlib import colors
from pathlib import Path
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
import math
import random

import inspect
import os
from datetime import datetime

import cv2
from tqdm.auto import tqdm

def polar_distance(radius, angle1, angle2) -> float:
    if angle2 < 0 or angle1 == angle2: return math.inf #prevent false positives
    radiant1, radiant2 = angle1 * math.pi / 180, angle2 * math.pi / 180
    return math.sqrt( 2 * radius**2 - 2 * radius**2 * math.cos( radiant1 - radiant2) )

def euclidean_distance(c1, c2) -> float:
    if any([c1[0],c1[1],c2[0], c2[1]]) < 0: return math.inf
    return math.sqrt( (c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 )

class DropoutType():
    NORMAL = 0
    BRIGHTNESS = 1

class Direction():
    HORIZONTAL = 0
    VERTICAL = 1

def dropout(d, d_first, value, dropout_p, dropout_type, max_dotsize) -> bool:
    dropout_p = 1 - dropout_p
    if dropout_p == 0: dropout_p += 10e-17

    avg_weighting = (max_dotsize / 2)**2
    if dropout_type == DropoutType.BRIGHTNESS: weighting =  (max_dotsize * (255-value) / 255)**2
    elif dropout_type == DropoutType.NORMAL: weighting = avg_weighting   # no affect from color darkness

    d += weighting
    d_first += weighting

    threshold = np.random.geometric(p=dropout_p) - 1
    return d < threshold + avg_weighting or d_first < threshold  + avg_weighting

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
    if folder and not folder.is_dir(): 
        folder.mkdir(parents=True, exist_ok=True)
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
    metadata.add_text('shape', str(img.size))
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
        print(type(data[key]))
        print(f'    {key}: ', data[key])
    print('')
    return data

class PrintableLambda( object ):
    def __init__( self, body ):
        self.body= body
    def __call__( self ):
        return eval( self.body )
    def __str__( self ):
        return self.body

# if result folder is specified: path names must have following pattern: *_{number}.* or no number containing
def build_video(source_folder:Path or str=None, image_paths:List[Path or str]=None, result_folder:Path or str='video_results', video_name:str='example', fps=25, max_quality=1080, file_format='png', sort_order=False) -> None:
    if type(source_folder) == str: source_folder = Path(source_folder)
    if type(result_folder) == str: result_folder = Path(result_folder)
    if not result_folder.is_dir():
        result_folder.mkdir(parents=True, exist_ok=True)

    def comparison_key(path:Path):
        name = path.stem
        splitted = name.split('_')
        if not re.findall('\d', name): return (name, 0)
        return (splitted[:-1], int(splitted[-1]))
    
    if not image_paths: 
        image_paths = sorted(source_folder.glob(f'*.{file_format}'), key=comparison_key, reverse=sort_order)

    if not '.' in video_name: video_name += '.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_name =str(result_folder / video_name)

    frame = cv2.imread(str(image_paths[0]))
    height, width, layers = frame.shape
    print(frame.shape)

    if height > max_quality or width > max_quality:
        if height > width:
            width = round(width * max_quality / height)
            height = max_quality
        else:
            height = round(height * max_quality / width)
            width = max_quality

    #print(width, height)
    #for path in image_paths:
    #    print(path.stem)

    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for img_path in tqdm(image_paths):
        img = Image.open(img_path)
        img.thumbnail((width, height))
        img = np.array(img, dtype=np.uint8)[:, :, :3] # rgba -> rgb
        img = img[:,:,::-1] # rgb -> bgr

        video.write(img)

    cv2.destroyAllWindows()
    video.release()
    print('Video created!')


if __name__ == '__main__':
    pass