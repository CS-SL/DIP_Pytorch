# make gif
import imageio
from PIL import Image 
import numpy as np 
import os

def make_gif(img_name, save_path, input_path, output_size=(800, 600)):
    imgs = []
    filelist = sorted(os.listdir(input_path))
    for i in filelist[:200]:
        if i.endswith('.png'):
            img = Image.open('{}/{}'.format(input_path, i))
            img = img.resize(output_size)
            img = np.array(img)
            imgs.append(img)

    output_file = os.path.join(save_path, '{}.gif'.format(img_name))
    imageio.mimsave(output_file, imgs, fps=20, duration=0.3)
    print('Done!')

if __name__ == "__main__":
    img_name = 'face-SR'
    save_path = '/root/proj/SelfSR/gif-results'
    input_path = '/root/proj/SelfSR/results/Set5-Face-linear-sc-tv+sr/SR/X4'
    make_gif(img_name, save_path, input_path, (512, 512))