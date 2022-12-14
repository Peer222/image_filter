from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import argparse
import math
from tqdm.auto import tqdm


import utils

MAX_IMAGE_SIZE = 2000 

# downsamples images automatically to 2000 as maximum edge size
def vertical_line_image(result_folder=None, img_path=None, img:Image.Image=None, background_color=255, line_width=8, line_window=4, space=2, line_color='original') -> Image.Image:
    arguments = locals().copy()
    result_folder, img_path = utils.get_filepaths(result_folder, img_path)

    white = 255
    centered = line_window // 2

    if img_path: img = Image.open(img_path)

    img = utils.downsample(img, MAX_IMAGE_SIZE)

    img = img.convert('RGB')
    img = np.array(img)

    background_color = utils.parse_color_format(background_color)

    line_img = np.full((img.shape[0], img.shape[1], 4), background_color, np.uint8)

    line_color = np.array(utils.parse_color_format(line_color))

    for x in tqdm(range(0, img.shape[0], 1)):
        y = 0
        while y <img.shape[1]:
            if x + centered >= img.shape[0] or y + centered >= img.shape[1]: break

            x_ = min(x + line_window, img.shape[0])
            y_ = min(y + line_window, img.shape[1])

            avg = np.mean(img[x:x_, y:y_], (0, 1))

            if line_color.any(): avg = (avg + line_color[:3]) / 2

            upper = min(y + centered + line_width//2, img.shape[1] - 1)
            lower = y + centered - line_width//2

            line_img[x + centered, lower:upper] = np.append(avg, 255)

            y += line_width + space

    return utils.save_image(result_folder, line_img, img_path, arguments)


def horizontal_line_image(result_folder=None, img_path=None, img:Image.Image=None, background_color=255, line_width=8, line_window=4, space=2, line_color='original') -> Image.Image:
    arguments = locals().copy()
    result_folder, img_path = utils.get_filepaths(result_folder, img_path)

    if img_path: img = Image.open(img_path)

    img = img.rotate(90, expand=True)

    img = vertical_line_image(img=img, background_color=background_color, line_width=line_width, line_window=line_window, space=space, line_color=line_color)

    img = img.rotate(-90, expand=True)

    return utils.save_image(result_folder, img, img_path, arguments)


# image is downsampled automatically
def diagonal_line_image(result_folder=None, img_path=None, img:Image.Image=None, background_color=255, 
                        line_width=8, line_window=4, space=0, angle=235, line_color='original') -> Image.Image:
    arguments = locals().copy()
    result_folder, img_path = utils.get_filepaths(result_folder, img_path)

    if img_path: img = Image.open(img_path)
    original_shape = img.size

    img = img.rotate(angle, expand=True)

    intermediate_shape = img.size

    img = vertical_line_image(img=img, background_color=background_color, line_width=line_width, line_window=line_window, space=space, line_color=line_color)

    img = img.rotate(-angle, expand=False)

    x_offset, y_offset = (intermediate_shape[0] - original_shape[0]) // 2, (intermediate_shape[1] - original_shape[1]) // 2
    scale = lambda x : int( x * img.size[0] / intermediate_shape[0])
    left, top, right, bottom = scale(x_offset), scale(y_offset), scale(x_offset + original_shape[0]), scale(y_offset + original_shape[1])
    img = img.crop((left, top, right, bottom))

    return utils.save_image(result_folder, img, img_path, arguments)


def experimental_line_image(result_folder=None, img_path=None, img:Image.Image=None, background_color=255, line_color=0, line_width=4, line_window=8, space=8) -> Image.Image:
    arguments = locals().copy()
    result_folder, img_path = utils.get_filepaths(result_folder, img_path)

    white = 255
    centered = line_window // 2

    if img_path: img = Image.open(img_path)

    ratio = min(1000/img.size[0], 1000/img.size[1])
    new_size = (int(img.size[0] * ratio),  int(img.size[1] * ratio))
    img.thumbnail(new_size)

    img = img.convert('L')
    img = np.array(img)

    background_color = utils.parse_color_format(background_color)
    line_color = utils.parse_color_format(line_color)

    line_img = np.full((img.shape[0], img.shape[1], 4), background_color, np.uint8)

    for x in tqdm(range(0, img.shape[0], 1)):
        y = 0
        while y < img.shape[1]:
            if x + centered >= img.shape[0] or y + centered >= img.shape[1]: break

            x_ = min(x + line_window, img.shape[0])
            y_ = min(y + line_window, img.shape[1])

            avg = np.mean(img[x:x_, y:y_], (0, 1))

            upper = int(min(y + centered + line_width//2, img.shape[1] - 1) * (255 - avg) / white)
            lower = int((y + centered - line_width//2) * (255 - avg) / white)

            line_img[x + centered, lower:upper] = line_color

            y += line_width

    return utils.save_image(result_folder, line_img, img_path, arguments)


def experimental_line2(result_folder=None, img_path=None, img:Image.Image=None, background_color=255, line_color=0, line_width=4, line_window=8, space=8) -> Image.Image:
    arguments = locals().copy()
    result_folder, img_path = utils.get_filepaths(result_folder, img_path)

    white = 255
    centered = line_window // 2

    if img_path: img = Image.open(img_path)

    ratio = min(1000/img.size[0], 1000/img.size[1])
    new_size = (int(img.size[0] * ratio),  int(img.size[1] * ratio))
    img.thumbnail(new_size)

    img = img.convert('L')
    img = np.array(img)

    background_color = utils.parse_color_format(background_color)
    line_color = utils.parse_color_format(line_color)

    line_img = np.full((img.shape[0], img.shape[1], 4), background_color, np.uint8)

    for x in tqdm(range(0, img.shape[0], 1)):
        y = 0
        while y < img.shape[1]:
            if x + centered >= img.shape[0] or y + centered >= img.shape[1]: break
            x_ = min(x + line_window, img.shape[0])
            y_ = min(y + line_window, img.shape[1])

            avg = np.mean(img[x:x_, y:y_], (0, 1))

            upper = int(min(y + centered + line_width//2, img.shape[1] - 1) * (255 - avg) / white)
            lower = int((y + centered - line_width//2) * (255 - avg) / white)

            line_img[x + centered, lower:upper] = line_color

            y += line_width

    return utils.save_image(result_folder, line_img, img_path, arguments)


def abstract_line_image(result_folder=None, img_path=None, img:Image.Image=None, background_color=255, line_color=0, 
                    max_line_width=16, direction=utils.Direction.HORIZONTAL, random_line_color=0.0) -> Image.Image:
    arguments = locals().copy()
    result_folder, img_path = utils.get_filepaths(result_folder, img_path)

    if img_path: img = Image.open(img_path)

    grayscale_img = img.convert('L')
    img, grayscale_img = np.array(img), np.array(grayscale_img)

    background_color = utils.parse_color_format(background_color)
    line_color = utils.parse_color_format(line_color)

    line_img = np.full((img.shape[0], img.shape[1], 4), background_color, np.uint8)

    white = 255
    centered = max_line_width // 2

    for x1 in tqdm(range(0, grayscale_img.shape[direction], max_line_width)):

        for x2 in range(0, grayscale_img.shape[1 - direction], 1):

            x1_ = min(x1 + max_line_width, grayscale_img.shape[direction])
            x2_ = min(x2 + max_line_width, grayscale_img.shape[1 - direction])

            if direction == utils.Direction.HORIZONTAL: avg = np.mean(grayscale_img[x1:x1_, x2:x2_], (0, 1))
            elif direction == utils.Direction.VERTICAL: avg = np.mean(grayscale_img[x2:x2_, x1:x1_], (0, 1))

            width = max(centered - round(avg / white * centered), 0)
        
            if line_color: color = line_color
            else: color = np.append(img[min(x1 + centered, img.shape[0]-1), min(x2 + centered, img.shape[1]-1)], 255).tolist()

            if random_line_color > 0: color = utils.get_random_color(line_color, spread=random_line_color)

            if direction == utils.Direction.HORIZONTAL: line_img[x1-width:x1+width, x2-centered:x2+centered] = color
            elif direction == utils.Direction.VERTICAL: line_img[x2-centered:x2+centered, x1-width:x1+width] = color


    return utils.save_image(result_folder, line_img, img_path, arguments)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='line image converter')

    abstract_line_image('results', 'contrast.png', None, (241, 255, 172), (0, 115, 151), 64, utils.Direction.VERTICAL)
    #diagonal_line_image('results', img_path='contrast.png', line_width=16)
    #vertical_line_image('results', 'contrast.png', line_color=(112, 166, 255))
