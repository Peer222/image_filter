from tqdm.auto import tqdm
from PIL import Image

import utils
import line_image_converter as lic
import dot_image_converter as dic

def video_test():
    #for step in tqdm(range(6000, 0, -5)):
    #    dic.custom_function_dot_image(result_folder='video_test', img_path='contrast.png', function=utils.PrintableLambda('lambda c: (c[0], c[1] + math.sin(c[0]/100) * 200)'),
    #                            dot_color=['59', '21', '28'], background_color=['231', '208', '213'], dot_distance=20, max_dotsize=25, spacing=2,
    #                            step_size=(0.1, step), domain=[0, 6048, -1000, 5000], random_dot_color=0.2)

    utils.build_video('video_test', None, video_name='test_video', fps=25)

def video_test2():
    img = Image.open('desierto.png')
    #for step in tqdm(range(0, 600, 1)):
    #    dic.custom_function_dot_image(result_folder='video_test2', img=img, function=utils.PrintableLambda(f'lambda c: (c[0], c[1] + math.sin(c[0]/100 + {step/100}) * 200)'),
    #    dot_color='black', background_color=[234, 197, 97], dot_distance=20, max_dotsize=25, spacing=2, step_size=(0.1, 70), domain=[0, 3906, 240, 5614])

    utils.build_video('video_test2', None, video_name='test_video2', fps=25)

def video_test3():
    img = Image.open('desierto.png')
    for step in tqdm(range(0, 600, 1)):
        dic.custom_function_dot_image(result_folder='video_test3', img=img, function=utils.PrintableLambda(f'lambda c: (c[0], c[1] + math.sin(c[0]/100 + {step/15}) * 200)'),
        dot_color='black', background_color=[234, 197, 97], dot_distance=20, max_dotsize=25, spacing=2, step_size=(0.1, 62), domain=[0, 3906, 240, 5614])

    utils.build_video('video_test3', None, video_name='test_video3', fps=25)

def video_test4():
    img = Image.open('desierto.png')
    for step in tqdm(range(0, 600, 1)):
        b_color = [int(234 + step * (255-234) / 600), int(197 + step * (255-197) / 600), int(97 + step * (255-97) / 600)]
        dic.custom_function_dot_image(result_folder='video_test4', img=img, function=utils.PrintableLambda(f'lambda c: (c[0] + math.sin(c[1]/200 + {step/25}) * 100, c[1])'),
        dot_color='black', background_color=b_color, dot_distance=20, max_dotsize=25, spacing=2, step_size=(62, 0.5), domain=[-380, 4256, 0, 5844])

    utils.build_video('video_test4', None, video_name='test_video4', fps=25)


if __name__ == '__main__':
    video_test4()