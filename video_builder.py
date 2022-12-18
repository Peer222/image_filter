from tqdm.auto import tqdm

import utils
import line_image_converter as lic
import dot_image_converter as dic

def video_test():
    #for step in tqdm(range(6000, 0, -5)):
    #    dic.custom_function_dot_image(result_folder='video_test', img_path='contrast.png', function=utils.PrintableLambda('lambda c: (c[0], c[1] + math.sin(c[0]/100) * 200)'),
    #                            dot_color=['59', '21', '28'], background_color=['231', '208', '213'], dot_distance=20, max_dotsize=25, spacing=2,
    #                            step_size=(0.1, step), domain=[0, 6048, -1000, 5000], random_dot_color=0.2)

    utils.build_video('video_test', None, video_name='test_video', fps=25)

if __name__ == '__main__':
    video_test()

    #utils.get_metadata('video_test/contrast_result_4.png')
    #utils.get_metadata('results/contrast_result_198.png')