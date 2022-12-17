from PIL import Image, ImageOps
import numpy as np
import cv2
from pathlib import Path
import argparse
import math
from timeit import default_timer as timer
import random

import utils

def gap_aware_polar_dot_image(result_folder=None, img_path=None, img:Image.Image=None, background_color=255, dot_color=0, 
                    max_dotsize=15, spacing=7, center=None, dropout=0.0, random_dot_color=0.0) -> Image.Image:
    arguments = locals().copy()
    result_folder, img_path = utils.get_filepaths(result_folder, img_path)

    if img_path: img = Image.open(img_path)

    grayscale_img = img.convert('L')
    img, grayscale_img = np.array(img), np.array(grayscale_img)

    background_color = utils.parse_color_format(background_color)
    dot_color = utils.parse_color_format(dot_color)

    dotted_img = np.full((img.shape[0], img.shape[1], 4), background_color, np.uint8)

    white = 255
    centered = max_dotsize // 2
    dropout = 1 - dropout
    if dropout == 0: dropout += 10e-17

    if not center: center = ( img.shape[0]//2, img.shape[1]//2 )
    max_radius = int(math.sqrt(img.shape[0]**2 + img.shape[1]**2))
    window_size = max_dotsize//2

    def polar_distance(radius, angle1, angle2):
        if angle2 < 0 or angle1 == angle2: return math.inf #prevent false positives
        radiant1, radiant2 = angle1 * math.pi / 180, angle2 * math.pi / 180
        return math.sqrt( 2 * radius**2 - 2 * radius**2 * math.cos( radiant1 - radiant2) )

    
    for radius in range(0, max_radius, max_dotsize):
        last_angle = -1

        perimeter = 2 * radius * math.pi
        num_dots = max(int(perimeter / max_dotsize), 1)

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

            # dropout
            d = polar_distance(radius, angle, last_angle)
            d_first = polar_distance(radius, angle, angle_offset)

            threshold = np.random.geometric(p=dropout) - 1
            if d < threshold or d_first < threshold: continue
            last_angle = angle

            if dot_color: color = dot_color
            else: color = np.append(img[x_, y_], 255).tolist()

            if random_dot_color > 0: color = utils.get_random_color(dot_color, spread=random_dot_color)

            cv2.circle(dotted_img, (y_, x_), dot_radius, color, -1)

    return utils.save_image(result_folder, dotted_img, img_path, arguments)

# recommended dropout: default or ~0.99
def polar_dot_image(result_folder=None, img_path=None, img:Image.Image=None, background_color=255, dot_color=0, 
                    max_dotsize=15, spacing=7, center=None, dropout=0.0, random_dot_color=0.0) -> Image.Image:
    arguments = locals().copy()
    result_folder, img_path = utils.get_filepaths(result_folder, img_path)

    white = 255
    centered = max_dotsize // 2
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

    window_size = max_dotsize//2

    def polar_distance(radius, angle1, angle2):
        if angle2 == None or angle1 == angle2: return math.inf #prevent false positives
        radiant1, radiant2 = angle1 * math.pi / 180, angle2 * math.pi / 180
        return math.sqrt( 2 * radius**2 - 2 * radius**2 * math.cos( radiant1 - radiant2) )

    for radius in range(0, max_radius, max_dotsize):
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

            # dropout
            d = polar_distance(radius, angle, last_angle)
            d_first = polar_distance(radius, angle, first_angle)

            threshold = np.random.geometric(p=dropout) - 1
            if d < threshold or d_first < threshold: continue
            last_angle = angle

            if dot_color: color = dot_color
            else: color = np.append(img[x_, y_], 255).tolist()

            if random_dot_color > 0: color = utils.get_random_color(dot_color, spread=random_dot_color)

            cv2.circle(dotted_img, (y_, x_), dot_radius, color, -1)

    return utils.save_image(result_folder, dotted_img, img_path, arguments)


# if result_folder == None -> image is not saved
def raster_dot_image(result_folder=None, img_path=None, img:Image.Image=None, background_color=255, dot_color=0, max_dotsize=15, spacing=0, random_dot_color=0.0) -> Image.Image:
    arguments = locals().copy()
    result_folder, img_path = utils.get_filepaths(result_folder, img_path)

    white = 255
    centered = max_dotsize // 2

    if img_path: img = Image.open(img_path)

    grayscale_img = img.convert('L')
    img, grayscale_img = np.array(img), np.array(grayscale_img)

    background_color = utils.parse_color_format(background_color)
    dot_color = utils.parse_color_format(dot_color)

    dotted_img = np.full((img.shape[0], img.shape[1], 4), background_color, np.uint8)

    for x in range(0, grayscale_img.shape[0], max_dotsize):

        for y in range(0, grayscale_img.shape[1], max_dotsize):

            x_ = min(x + max_dotsize, grayscale_img.shape[0])
            y_ = min(y + max_dotsize, grayscale_img.shape[1])
            avg = np.mean(grayscale_img[x:x_, y:y_], (0, 1))

            radius = max(centered - round(avg / white * centered) - spacing, 0)
        
            if dot_color: color = dot_color
            else: color = np.append(img[min(x + centered, img.shape[0]-1), min(y + centered, img.shape[1]-1)], 255).tolist()

            if random_dot_color > 0: color = utils.get_random_color(dot_color, spread=random_dot_color)

            cv2.circle(dotted_img, (y + centered, x + centered), radius, color, -1)

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
    img, grayscale_img = np.array(img), np.array(grayscale_img)

    background_color = utils.parse_color_format(background_color)
    dot_color = utils.parse_color_format(dot_color)

    dotted_img = np.full((img.shape[0], img.shape[1], 4), background_color, np.uint8)

    values = []

    window_size = 2

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
    parser = argparse.ArgumentParser(description='dot_image')

    parser.add_argument('-i', '--image', required=True, type=str, help='file path of input image')
    parser.add_argument('--background', default=255, nargs="*", help='numeric background color')
    parser.add_argument('--dot_color', default=0, nargs="*", help='dot color or original')
    parser.add_argument('--dot_size', default=25, help='maximum dot size in pixel')
    parser.add_argument('--dot_spacing', default=0, help='reduces the dot size without reducing the dot-calculation window')

    args = parser.parse_args()

    #raster_dot_image(result_folder=Path('results'), img_path=args.image, background_color=args.background, dot_color=args.dot_color, max_dotsize=args.dot_size, spacing=args.dot_spacing)
    #random_dot_image(result_folder=Path('results'), img_path=args.image, background_color=args.background, dot_color=args.dot_color)

    gap_aware_polar_dot_image(result_folder=Path('results'), img_path=args.image, background_color=args.background, dot_color=args.dot_color, max_dotsize=args.dot_size, spacing=args.dot_spacing)#, dropout=0.99, random_dot_color=0.3)

    #utils.get_metadata('results/contrast_result_77.png')

    print('Dot image successfully created!')