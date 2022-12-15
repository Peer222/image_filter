from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import argparse
import math
from tqdm.auto import tqdm


import utils

def vertical_line_image(result_folder=None, img_path=None, img:Image.Image=None, background_color=255, line_color=0, line_width=2, line_window=2, space=4) -> Image.Image:
    result_folder, img_path = utils.get_filepaths(result_folder, img_path)

    white = 255
    centered = line_window // 2

    if img_path: img = Image.open(img_path)

    ratio = min(1000/img.size[0], 1000/img.size[1])
    new_size = (int(img.size[0] * ratio),  int(img.size[1] * ratio))
    img.thumbnail(new_size)

    img = img.convert('RGB')
    img = np.array(img)

    background_color = utils.parse_color_format(background_color)

    line_img = np.full((img.shape[0], img.shape[1], 4), background_color, np.uint8)

    for x in tqdm(range(0, img.shape[0], 1)):
        y = 0
        while y <img.shape[1]:
            if x + centered >= img.shape[0] or y + centered >= img.shape[1]: break
            #x_ = min(x + line_window, img.shape[0])
            #y_ = min(y + line_window, img.shape[1])

            #avg = np.sum(img[x:x+x_, y:y+y_], (0, 1))

            #avg = avg // line_window**2
            avg = np.array([0, 0, 0])
            for x_ in range(x, min(x + line_window, img.shape[0])):
                for y_ in range(y, min(y + line_window, img.shape[1])):
                    avg += img[x_][y_]
            avg = avg // line_window**2

            upper = min(y + centered + line_width//2, img.shape[1] - 1)
            lower = y + centered - line_width//2

            line_img[x + centered, lower:upper] = np.append(avg, 255)

            y += line_width + space

    final_img = Image.fromarray(line_img)
    if result_folder and img_path: final_img.save(result_folder / f"{img_path.stem}_{line_window}-{line_width}-{space}_result.png")
    elif result_folder: final_img.save(result_folder / f"horizontal_line_image_{line_window}-{line_width}-{space}_result.png")
    return final_img

def horizontal_line_image(result_folder=None, img_path=None, img:Image.Image=None, background_color=255, line_color=0, line_width=4, line_window=8, space=8) -> Image.Image:
    result_folder, img_path = utils.get_filepaths(result_folder, img_path)

    white = 255
    centered = line_window // 2

    if img_path: img = Image.open(img_path)

    ratio = min(1000/img.size[0], 1000/img.size[1])
    new_size = (int(img.size[0] * ratio),  int(img.size[1] * ratio))
    img.thumbnail(new_size)

    img = img.convert('RGB')
    img = np.array(img)

    background_color = utils.parse_color_format(background_color)

    line_img = np.full((img.shape[0], img.shape[1], 4), background_color, np.uint8)

    for y in tqdm(range(0, img.shape[1], 1)):
        x = 0
        while x < img.shape[0]:
            if x + centered >= img.shape[0] or y + centered >= img.shape[1]: break

            avg = np.array([0, 0, 0])
            for x_ in range(x, min(x + line_window, img.shape[0])):
                for y_ in range(y, min(y + line_window, img.shape[1])):
                    avg += img[x_][y_]
            avg = avg // line_window**2

            upper = min(x + centered + line_width//2, img.shape[0] - 1)
            lower = x + centered - line_width//2

            line_img[lower:upper, y + centered] = np.append(avg, 255)

            x += line_width + space

    final_img = Image.fromarray(line_img)
    if result_folder and img_path: final_img.save(result_folder / f"{img_path.stem}_{line_window}-{line_width}-{space}_result.png")
    elif result_folder: final_img.save(result_folder / f"horizontal_line_image_{line_window}-{line_width}-{space}_result.png")
    return final_img

def test(result_folder=None, img_path=None, img:Image.Image=None, background_color=255, line_color=0, line_width=4, line_window=8, space=8) -> Image.Image:
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
        while y <img.shape[1]:
            if x + centered >= img.shape[0] or y + centered >= img.shape[1]: break
            #x_ = min(x + line_window, img.shape[0])
            #y_ = min(y + line_window, img.shape[1])

            #avg = np.sum(img[x:x+x_, y:y+y_], (0, 1))

            #avg = avg // line_window**2
            avg = 0
            for x_ in range(x, min(x + line_window, img.shape[0])):
                for y_ in range(y, min(y + line_window, img.shape[1])):
                    avg += img[x_][y_]
            avg = avg // line_window**2

            upper = int(min(y + centered + line_width//2, img.shape[1] - 1) * (255 - avg) / white)
            lower = int((y + centered - line_width//2) * (255 - avg) / white)

            line_img[x + centered, lower:upper] = line_color

            y += line_width

    final_img = Image.fromarray(line_img)
    if result_folder and img_path: final_img.save(result_folder / f"{img_path.stem}_{line_window}-{line_width}-{space}_result.png")
    elif result_folder: final_img.save(result_folder / f"horizontal_line_image_{line_window}-{line_width}-{space}_result.png")
    return final_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='line image converter')

    test('results', 'contrast.png', None, 255, 0)

