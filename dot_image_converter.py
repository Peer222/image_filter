from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import argparse

import utils

# if result_folder == None -> image is not saved
def raster_dot_image(result_folder=None, img_path=None, img:Image.Image=None, background_color=255, dot_color=0, max_dotsize=15, spacing=0) -> Image.Image:
    if img_path and type(img_path) == str: img_path = Path(img_path)
    if type(result_folder) == str: result_folder = Path(result_folder)
    if result_folder and not result_folder.is_dir(): result_folder.mkdir(parents=True, exist_ok=True)

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
            avg = 0
            for x_ in range(x, min(x + max_dotsize, grayscale_img.shape[0])):
                for y_ in range(y, min(y + max_dotsize, grayscale_img.shape[1])):
                    avg += grayscale_img[x_][y_]
            avg = round(avg / max_dotsize**2)

            radius = max(centered - int(round(avg / white * centered)) - spacing, 0)
        
            if dot_color: color = dot_color
            else: color = np.append(img[min(x + centered, img.shape[0]-1), min(y + centered, img.shape[1]-1)], 255).tolist()
            cv2.circle(dotted_img, (y + centered, x + centered), radius, color, -1)


    final_img = Image.fromarray(dotted_img)
    if result_folder and img_path: final_img.save(result_folder / f"{img_path.stem}_{max_dotsize}-{spacing}-{dot_color}-{background_color}_result.png")
    elif result_folder: final_img.save(result_folder / f"raster_dot_image_{max_dotsize}-{spacing}-{dot_color}-{background_color}_result.png")
    return final_img

# prob around 0.00001
def random_dot_image(result_folder=None, img_path=None, img:Image.Image=None, prob=0.00001, 
                    background_color=255, dot_color=0, num_dots:float=0.1, dot_size=1) -> Image.Image:
    if img_path and type(img_path) == str: img_path = Path(img_path)
    if type(result_folder) == str: result_folder = Path(result_folder)
    if result_folder and not result_folder.is_dir(): result_folder.mkdir(parents=True, exist_ok=True)

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

            value = 0
            for x_ in range(x, x + window_size):
                if x_ >= grayscale_img.shape[0]: break
                for y_ in range(y, y + window_size):
                    if y_ >= grayscale_img.shape[1]: break
                    value += grayscale_img[x_][y_]
            values.append([x, y, value])
            
    values.sort(key=lambda v: v[2]) # sort by value

    num_dots = int( num_dots * img.shape[0] * img.shape[1] ) * 5
    i = 0
    while i < num_dots:
        pos = np.random.geometric(prob) - 1
        if pos >= len(values): continue

        x, y, _ = values[pos]
        #cv2.circle(dotted_img, (y, x), dot_size, dot_color, -1)
        if dot_color: dotted_img[x, y] = dot_color
        else: dotted_img[x, y] = np.append(img[x, y], 255)

        i += 1

    final_img = Image.fromarray(dotted_img)
    if result_folder and img_path: final_img.save(result_folder / f"{img_path.stem}_{prob}-{num_dots}-{dot_size}-{dot_color}-{background_color}_result.png")
    elif result_folder: final_img.save(result_folder / f"random_dot_image_{prob}-{num_dots}-{dot_size}-{dot_color}-{background_color}_result.png")
    return final_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dot_image')

    parser.add_argument('-i', '--image', required=True, type=str, help='file path of input image')
    parser.add_argument('--background', default=255, nargs="*", help='numeric background color')
    parser.add_argument('--dot_color', default=0, nargs="*", help='dot color or original')
    parser.add_argument('--dot_size', default=25, help='maximum dot size in pixel')
    parser.add_argument('--dot_spacing', default=0, help='reduces the dot size without reducing the dot-calculation window')

    args = parser.parse_args()

    raster_dot_image(result_folder=Path('results'), img_path=args.image, background_color=args.background, dot_color=args.dot_color, max_dotsize=args.dot_size, spacing=args.dot_spacing)
    #random_dot_image(result_folder=Path('results'), img_path=args.image, background_color=args.background, dot_color=args.dot_color)

    print('Dot image successfully created!')