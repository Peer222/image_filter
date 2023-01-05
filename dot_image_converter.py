from PIL import Image, ImageOps
from typing import Tuple
import numpy as np
import cv2
from pathlib import Path
import argparse
import math
import random
from tqdm.auto import tqdm

from timeit import default_timer as timer

import torch

import utils

# lambda c: (c[0] + 20 * math.sin(c[1]), c[1] + 20 * math.cos(c[0]))
# lambda c: (c[0] + math.sin(c[0]/20) * 30, 0.05 * (c[1]+2000) * (c[0]+ 1000))
# lambda c: (c[0], c[1] + math.sin(c[0]) * 200) (1, 300)
# lambda c: (c[0], c[1] + math.sin(c[0]/100) * 200) (0.1, 100)
def custom_function_dot_image(result_folder=None, img_path=None, img:Image.Image=None, function=utils.PrintableLambda('lambda c: (c[0], c[1] + math.sin(c[0]/100) * 200)'), 
                            dot_distance=15, background_color=255, dot_color=0, dot_size=15, spacing=7, timing='pre', step_size=(0.1, 10),
                            domain:Tuple[int, int, int, int]=None, # if None image_boundaries are used
                            dropout=0.0, dropout_type=utils.DropoutType.NORMAL, random_dot_color=0.0) -> Image.Image:
    arguments = locals().copy()
    function = function()
    result_folder, img_path = utils.get_filepaths(result_folder, img_path)

    if img_path: img = Image.open(img_path)

    grayscale_img = img.convert('L')
    img, grayscale_img = np.array(img), np.array(grayscale_img)

    background_color = utils.parse_color_format(background_color)
    dot_color = utils.parse_color_format(dot_color)

    dotted_img = np.full((img.shape[0], img.shape[1], 4), background_color, np.uint8)

    white = 255
    centered = dot_size // 2
    if not domain: domain = [0, grayscale_img.shape[0], 0, grayscale_img.shape[1]]
    if len(domain) != 4: raise ValueError('invalid domain boundaries, correct format: [x_lower, x_upper, y_lower, y_upper]')

    last_coordinates2 = (-1, -1)
    for x in np.arange(domain[0], domain[1], step_size[0]):
        x_transformed, _ = function((x, 0))
        if utils.euclidean_distance((x_transformed, 0), last_coordinates2) <= dot_distance: continue # without producing lines

        last_coordinates2 = (x_transformed, 0)
        last_coordinates = (-1, -1)

        for y in np.arange(domain[2], domain[3], step_size[1]):

            if timing == 'pre': x_transformed, y_transformed = function((x, y))
            else: x_transformed, y_transformed = x, y

            d = utils.euclidean_distance((x_transformed, y_transformed), last_coordinates)

            x_transformed, y_transformed = int(x_transformed), int(y_transformed)
            if not(0 <= x_transformed < grayscale_img.shape[0]) or not(0 <= y_transformed < grayscale_img.shape[1]): continue
            if d <= dot_distance: continue

            x_ = min(x_transformed + dot_size, grayscale_img.shape[0])
            y_ = min(y_transformed + dot_size, grayscale_img.shape[1])
            avg = np.mean(grayscale_img[x_transformed:x_, y_transformed:y_], (0, 1))

            radius = max(centered - int(avg / white * centered) - spacing, 0)

            if utils.dropout(d, math.inf, avg, dropout, dropout_type, dot_size): continue
            last_coordinates = (x_transformed, y_transformed)
        
            if dot_color: color = dot_color
            else: color = np.append(img[min(x_transformed + centered, img.shape[0]-1), min(y_transformed + centered, img.shape[1]-1)], 255).tolist()

            if random_dot_color > 0: color = utils.get_random_color(dot_color, spread=random_dot_color)

            if timing == 'post': x_transformed, y_transformed = function((x, y))
            if not(0 <= x_transformed < grayscale_img.shape[0]) or not(0 <= y_transformed < grayscale_img.shape[1]): continue

            cv2.circle(dotted_img, (y_transformed + centered, x_transformed + centered), radius, color, -1)

    return utils.save_image(result_folder, dotted_img, img_path, arguments)

# recommended dropout: default or ~0.99
def gap_aware_polar_dot_image(result_folder=None, img_path=None, img:Image.Image=None, background_color=255, dot_color=0, 
                    dot_size=15, spacing=7, center=None, dropout=0.0, dropout_type=utils.DropoutType.NORMAL, random_dot_color=0.0) -> Image.Image:
    arguments = locals().copy()
    result_folder, img_path = utils.get_filepaths(result_folder, img_path)

    if img_path: img = Image.open(img_path)

    grayscale_img = img.convert('L')
    img, grayscale_img = np.array(img), np.array(grayscale_img)

    background_color = utils.parse_color_format(background_color)
    dot_color = utils.parse_color_format(dot_color)

    dotted_img = np.full((img.shape[0], img.shape[1], 4), background_color, np.uint8)

    white = 255
    centered = dot_size // 2

    if not center: center = ( img.shape[0]//2, img.shape[1]//2 )
    max_radius = int(math.sqrt(img.shape[0]**2 + img.shape[1]**2))
    window_size = dot_size//2
    
    for radius in range(0, max_radius, dot_size):
        last_angle = -1

        perimeter = 2 * radius * math.pi
        num_dots = max(int(perimeter / dot_size), 1)

        angle_offset = random.randint(0, 360)

        for i_dot in range(0, num_dots):
            angle = i_dot / num_dots * 360 + angle_offset
            radiant = angle * math.pi / 180
            
            x_ = int(radius * math.cos(radiant)) + center[0]
            y_ = int(radius * math.sin(radiant)) + center[1]

            if not( 0 <= x_ < img.shape[0] and 0 <= y_ < img.shape[1] ): continue

            dx1, dx2 = max(x_ - window_size, 0), min(x_ + window_size, img.shape[0])
            dy1, dy2 = max(y_ - window_size, 0), min(y_ + window_size, img.shape[1])
            avg = np.mean(grayscale_img[dx1:dx2, dy1:dy2], (0, 1))

            dot_radius = max(centered - round(avg / white * centered) - spacing, 0) 

            d = utils.polar_distance(radius, angle, last_angle)
            d_first = utils.polar_distance(radius, angle, angle_offset)

            if utils.dropout(d, d_first, avg, dropout, dropout_type, dot_size): continue
            last_angle = angle

            if dot_color: color = dot_color
            else: color = np.append(img[x_, y_], 255).tolist()

            if random_dot_color > 0: color = utils.get_random_color(dot_color, spread=random_dot_color)

            cv2.circle(dotted_img, (y_, x_), dot_radius, color, -1)

    return utils.save_image(result_folder, dotted_img, img_path, arguments)

# recommended dropout: default or ~0.99
def polar_dot_image(result_folder=None, img_path=None, img:Image.Image=None, background_color=255, dot_color=0, 
                    dot_size=15, spacing=7, center=None, dropout=0.0, dropout_type=utils.DropoutType.NORMAL, random_dot_color=0.0) -> Image.Image:
    arguments = locals().copy()
    result_folder, img_path = utils.get_filepaths(result_folder, img_path)

    white = 255
    centered = dot_size // 2
    dropout = 1 - dropout
    if dropout == 0: dropout += 10e-17

    if img_path: img = Image.open(img_path)

    grayscale_img = img.convert('L')
    img, grayscale_img = np.array(img), np.array(grayscale_img)

    background_color = utils.parse_color_format(background_color)
    dot_color = utils.parse_color_format(dot_color)

    dotted_img = np.full((img.shape[0], img.shape[1], 4), background_color, np.uint8)

    if not center: center = ( img.shape[0]//2, img.shape[1]//2 )

    max_radius = int(math.sqrt(img.shape[0]**2 + img.shape[1]**2))

    window_size = dot_size//2

    for radius in range(0, max_radius, dot_size):
        last_angle = None
        first_angle = random.randint(0, 360)
        for angle in range(first_angle, first_angle + 360, 1):
            radiant = angle * math.pi / 180

            x_ = int(radius * math.cos(radiant)) + center[0]
            y_ = int(radius * math.sin(radiant)) + center[1]

            if not( 0 <= x_ < img.shape[0] and 0 <= y_ < img.shape[1] ): continue

            dx1, dx2 = max(x_ - window_size, 0), min(x_ + window_size, img.shape[0])
            dy1, dy2 = max(y_ - window_size, 0), min(y_ + window_size, img.shape[1])
            avg = np.mean(grayscale_img[dx1:dx2, dy1:dy2], (0, 1))

            dot_radius = max(centered - round(avg / white * centered) - spacing, 0) 

            d = utils.polar_distance(radius, angle, last_angle)
            d_first = utils.polar_distance(radius, angle, first_angle)

            if utils.dropout(d, d_first, avg, dropout, dropout_type, dot_size): continue
            last_angle = angle

            if dot_color: color = dot_color
            else: color = np.append(img[x_, y_], 255).tolist()

            if random_dot_color > 0: color = utils.get_random_color(dot_color, spread=random_dot_color)

            cv2.circle(dotted_img, (y_, x_), dot_radius, color, -1)

    return utils.save_image(result_folder, dotted_img, img_path, arguments)


# if result_folder == None -> image is not saved
def raster_dot_image(result_folder=None, img_path=None, img:Image.Image=None, background_color=255, dot_color=0, dot_size=15, spacing=0, random_dot_color=0.0) -> Image.Image:
    arguments = locals().copy()
    result_folder, img_path = utils.get_filepaths(result_folder, img_path)

    white = 255
    centered = dot_size // 2

    if img_path: img = Image.open(img_path)

    grayscale_img = img.convert('L')
    
    img = np.array(img, dtype=np.uint8) #np.array(grayscale_img, dtype=np.uint8)
    grayscale_img = utils.transform(grayscale_img).to(torch.float32)
    grayscale_img.requires_grad = False
    #grayscale_img = np.array(grayscale_img, dtype=np.uint8)

    background_color = utils.parse_color_format(background_color)
    dot_color = utils.parse_color_format(dot_color)

    dotted_img = np.full((img.shape[0], img.shape[1], 4), background_color, np.uint8)

    # ~23% speedup against for-loop/np.mean combination (comment below)
    #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    convolution = torch.nn.Conv2d(1, 1, dot_size, stride=dot_size, padding=0, bias=False)#, padding_mode='replicate')
    with torch.no_grad():
        convolution.weight = torch.nn.Parameter(torch.ones_like(convolution.weight, dtype=torch.float32), requires_grad=False)
        filtered = convolution(grayscale_img).squeeze()

        for x in range(0, filtered.shape[0]):
            for y in range(0, filtered.shape[1]):
                
                radius = max(centered - round(filtered[x, y].item() / dot_size**2 / white * centered) - spacing, 0)
        
                if dot_color: color = dot_color
                else: color = np.append(img[min(x + centered, img.shape[0]-1), min(y + centered, img.shape[1]-1)], 255).tolist()

                if random_dot_color > 0: color = utils.get_random_color(dot_color, spread=random_dot_color)

                cv2.circle(dotted_img, (y * dot_size + centered, x * dot_size + centered), radius, color, -1)

    """
    for x in range(0, grayscale_img.shape[0], dot_size):

        for y in range(0, grayscale_img.shape[1], dot_size):

            x_ = min(x + dot_size, grayscale_img.shape[0])
            y_ = min(y + dot_size, grayscale_img.shape[1])
            avg = np.mean(grayscale_img[x:x_, y:y_], (0, 1))

            radius = max(centered - round(avg / white * centered) - spacing, 0)
        
            if dot_color: color = dot_color
            else: color = np.append(img[min(x + centered, img.shape[0]-1), min(y + centered, img.shape[1]-1)], 255).tolist()

            if random_dot_color > 0: color = utils.get_random_color(dot_color, spread=random_dot_color)

            cv2.circle(dotted_img, (y + centered, x + centered), radius, color, -1)
    """
    return utils.save_image(result_folder, dotted_img, img_path, arguments)

# prob around 0.00001
def random_dot_image(result_folder=None, img_path=None, img:Image.Image=None, prob=0.00001, 
                    background_color=255, dot_color=0, num_dots:float=0.1, dot_size=1, random_dot_color=0.0) -> Image.Image:
    arguments = locals().copy()
    result_folder, img_path = utils.get_filepaths(result_folder, img_path)

    if img_path: img = Image.open(img_path)

    # downsample image under specific size
    ratio = min(1000/img.size[0], 1000/img.size[1])
    new_size = (int(img.size[0] * ratio),  int(img.size[1] * ratio))
    img.thumbnail(new_size)

    grayscale_img = img.convert('L')
    img = np.array(img)
    grayscale_img = np.array(grayscale_img)

    background_color = utils.parse_color_format(background_color)
    dot_color = utils.parse_color_format(dot_color)

    dotted_img = np.full((img.shape[0], img.shape[1], 4), background_color, np.uint8)

    values = []

    window_size = 2

    # no significant speedup with torch convolution
    for x in range(0, grayscale_img.shape[0]):
        for y in range(0, grayscale_img.shape[1]):

            x_ = min(x + window_size, grayscale_img.shape[0])
            y_ = min(y + window_size, grayscale_img.shape[1])
            value = np.sum(grayscale_img[x:x_, y:y_], (0, 1))

            values.append([x, y, value])

    values.sort(key=lambda v: v[2]) # sort by value

    num_dots = int( num_dots * img.shape[0] * img.shape[1] ) * 5
    i = 0
    while i < num_dots:
        pos = np.random.geometric(prob) - 1

        if pos >= len(values): continue

        x, y, _ = values[pos]

        if dot_color: color = dot_color
        else: color = np.append(img[x, y], 255)

        if random_dot_color > 0: color = utils.get_random_color(dot_color, spread=random_dot_color)

        if dot_size <= 1: dotted_img[x, y] = color
        else: cv2.circle(dotted_img, (y, x), dot_size//2, color, -1)

        i += 1

    return utils.save_image(result_folder, dotted_img, img_path, arguments)


# color original does not work due to invertion
def reversed_random_dot_image(result_folder=None, img_path=None, img:Image.Image=None, prob=0.00001, 
                    background_color=0, dot_color=255, num_dots:float=0.1, dot_size=1, random_dot_color=0.0) -> Image.Image:
    arguments = locals().copy()
    result_folder, img_path = utils.get_filepaths(result_folder, img_path)

    if img_path: img = Image.open(img_path)
    img = ImageOps.invert(img)

    dotted_img = random_dot_image(img=img, prob=prob, background_color=background_color, dot_color=dot_color, num_dots=num_dots, dot_size=dot_size, random_dot_color=random_dot_color)

    return utils.save_image(result_folder, dotted_img, img_path, arguments)

if __name__ == '__main__':
    desc = """
    Creates different dot images depending on specified options.\n
    Options with abbreviations are applicable to all methods.\n\n

    List of available additional options for respective methods:\n
    raster_dot_image:\t                               spacing\n
    random_dot_image, reversed_random_dot_image:\t    num_dots, probability\n
    gap_aware_polar_dot_image, polar_dot_image:\t     spacing, dropout, dropout_type, center\n
    custom_function_dot_image:\t                      spacing, dropout, dropout_type, function, domain, step_size, dot_distance\n
    """
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-f', '--result_folder', default=Path("results"), type=Path)
    parser.add_argument('-i', '--img_path', required=True, type=str, help='file path of input image')
    parser.add_argument('-m', '--method', default="raster_dot_image", choices=["raster_dot_image", "random_dot_image", "polar_dot_image", "gap_aware_polar_dot_image", "reversed_random_dot_image", "custom_function_dot_image"])
    parser.add_argument('-b', '--background_color', default=255, nargs="*", help='background color or original')
    parser.add_argument('-c', '--dot_color', default=0, nargs="*", help='dot color or original')
    parser.add_argument('-s', '--dot_size', type=int, default=-1, help='maximum dot size in pixel, default = 15/1 depending on method')
    parser.add_argument('-r', '--random_dot_color', type=float, default=0.0)

    parser.add_argument('--spacing', type=int, default=7, help='Reduces the dot size without reducing the dot-calculation window')

    parser.add_argument('--dropout', type=float, default=0.0, help='~Probability~ of unpainted dots')
    parser.add_argument('--dropout_type', type=int, default=utils.DropoutType.NORMAL, choices=[utils.DropoutType.NORMAL, utils.DropoutType.BRIGHTNESS], help="Effects dropout probability")

    parser.add_argument('--center', type=Tuple[int], default=None, help="Position of center of the circle")

    parser.add_argument('--function', type=str, default='lambda c: (c[0], c[1] + math.sin(c[0]/100) * 200)', help="Lambda function which accepts (x, y) and returns (x_, y_)")
    parser.add_argument('--domain', type=Tuple[int, int, int, int], default=None, help="X and y ranges on which to calculate the custom function")
    parser.add_argument('--step_size', type=Tuple[float, float], default=(0.1, 10), help="frequency to calculate x and y respectively")
    parser.add_argument('--dot_distance', type=int, default=15, help="Minimum distance between dots")

    parser.add_argument('--num_dots', type=float, default=0.1, help="Fraction of number of pixels")
    parser.add_argument('--prob', type=float, default=0.00001)


    args = parser.parse_args()

    # (reversed)_random_dot_image
    if 'random' in args.method:
        if args.dot_size < 0: args.dotsize = 1
        eval(args.method)(args.result_folder, args.img_path, None, background_color=args.background_color, dot_color=args.dot_color, dot_size=args.dot_size, random_dot_color=args.random_dot_color, num_dots=args.num_dots, prob=args.prob)
    
    # custom function
    if 'custom' in args.method:
        if args.dot_size < 0: args.dotsize = 15
        eval(args.method)(args.result_folder, args.img_path, None, background_color=args.background_color, dot_color=args.dot_color, dot_size=args.dot_size, random_dot_color=args.random_dot_color, spacing=args.spacing, dropout=args.dropout, dropout_type=args.dropout_type, domain=args.domain, function=utils.PrintableLambda(args.function), step_size=args.step_size, dot_distance=args.dot_distance)

    # (gap_aware)_polar_dot_image
    if 'polar' in args.method:
        if args.dot_size < 0: args.dotsize = 15
        eval(args.method)(args.result_folder, args.img_path, None, background_color=args.background_color, dot_color=args.dot_color, dot_size=args.dot_size, random_dot_color=args.random_dot_color, spacing=args.spacing, dropout=args.dropout, dropout_type=args.dropout_type, center=args.center)

    # raster_dot_image
    if 'raster' in args.method:
        if args.dot_size < 0: args.dotsize = 15
        eval(args.method)(args.result_folder, args.img_path, None, background_color=args.background_color, dot_color=args.dot_color, dot_size=args.dot_size, random_dot_color=args.random_dot_color, spacing=args.spacing)

    #raster_dot_image(result_folder=Path('results'), img_path=args.image, background_color=args.background, dot_color=args.dot_color, dot_size=args.dot_size, spacing=args.dot_spacing)
    #random_dot_image(result_folder=Path('results'), img_path=args.image, background_color=args.background, dot_color=args.dot_color)

    #gap_aware_polar_dot_image(result_folder=Path('results'), img_path=args.image, background_color=args.background, dot_color=args.dot_color, dot_size=args.dot_size, spacing=args.dot_spacing, dropout=0.99, dropout_type=utils.DropoutType.NORMAL, random_dot_color=0.3)
    #custom_function_dot_image(result_folder=Path('results'), img_path=args.image, background_color=args.background, dot_color=args.dot_color, dot_size=args.dot_size, dot_distance=20, spacing=args.dot_spacing, dropout=0.99, dropout_type=utils.DropoutType.NORMAL, domain=[0, 6048, -1000, 5000])
    #utils.get_metadata('results/contrast_result_168.png')

    print('Dot image successfully created!')