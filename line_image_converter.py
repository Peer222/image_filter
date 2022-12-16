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
                        line_width=8, line_window=4, space=1, to_right=True, line_color='original') -> Image.Image:
    arguments = locals().copy()
    result_folder, img_path = utils.get_filepaths(result_folder, img_path)

    if img_path: img = Image.open(img_path)
    original_shape = img.size

    if to_right: img = img.rotate(-45, expand=True)
    else: img.rotate(45, expand=True)

    intermediate_size = img.size[0]

    img = vertical_line_image(img=img, background_color=background_color, line_width=line_width, line_window=line_window, space=space, line_color=line_color)

    if to_right: img = img.rotate(45, expand=False)
    else: img = img.rotate(-45, expand=False)

    x_offset, y_offset = (intermediate_size - original_shape[0]) // 2, (intermediate_size - original_shape[1]) // 2
    scale = lambda x : int( x * img.size[0] / intermediate_size)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='line image converter')

    #experimental_line2('results', 'contrast.png', None, (241, 255, 172), (0, 115, 151, 0))
    #diagonal_line_image('results', img_path='contrast.png', line_width=16)
    vertical_line_image('results', 'contrast.png', line_color=(112, 166, 255))
