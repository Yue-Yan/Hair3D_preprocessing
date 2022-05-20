import os
import cv2
import io
import requests
import numpy as np
from PIL import Image

# 分两步来完成
def call_super_resolution(colorImg_path):
    '''Send an input image through the network.'''
    model_endpoint = 'http://localhost:5000/model/predict'

    with open(colorImg_path, 'rb') as file:
        file_form = {'image': (colorImg_path, file, 'image/png')}
        r = requests.post(url=model_endpoint, files=file_form)
        assert r.status_code == 200
        im = Image.open(io.BytesIO(r.content)) # Type: numpy array, shape: 1800 X 1800

        super_save_root = os.path.dirname(colorImg_path)
        super_save_name = os.path.basename(colorImg_path).split('.')[0] + '_AAAAA.png'
        super_save_path = os.path.join(super_save_root, super_save_name)
        im.save(super_save_path)
    return super_save_path


def resize_1800_to_1024(super_save_path):
    super_img = cv2.imread(super_save_path)
    print('原图尺寸：', super_img.shape)
    cropped_super_img = cv2.resize(super_img, (1024,1024), interpolation=cv2.INTER_LINEAR)
    print('新图尺寸：', cropped_super_img.shape)
    cropped_super_root = os.path.dirname(super_save_path)
    cropped_super_name = os.path.basename(super_save_path).split('.')[0] + '_AAAAA_cropped.png'
    cropped_super_path = os.path.join(cropped_super_root, cropped_super_name)
    # print(type(cropped_super_img)) # numpy array
    cv2.imwrite(cropped_super_path, cropped_super_img)
    return None


# 其实，不需要两个函数，直接在1800的基础上直接resize成1024的就可以
# 下面代码跑通了，但是人脸是蓝色的
def call_super_resolution_directly(colorImg_path):
    '''Send an input image through the network.'''
    model_endpoint = 'http://localhost:5000/model/predict'

    with open(colorImg_path, 'rb') as file:
        file_form = {'image': (colorImg_path, file, 'image/png')}
        r = requests.post(url=model_endpoint, files=file_form)
        print('XXXXX', type(r.content))
        assert r.status_code == 200
        # im = Image.open(io.BytesIO(r.content)) # Type: numpy array, shape: 1800 X 1800
        im = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        # im = Image.fromarray(im) # 这句是用来把numpy array保存为图片的
        print('im type: ', type(im))
        # resize成1024
        cropped_im = cv2.resize(im, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        print('cropped: ', type(cropped_im))
        super_save_root = os.path.dirname(colorImg_path)
        super_save_name = os.path.basename(colorImg_path).split('.')[0] + '_AAAAA.png'
        super_save_path = os.path.join(super_save_root, super_save_name)
        # cropped_im.save(super_save_path) # 因为是Numpy array格式，所以要用cv2来保存
        # tmp = Image.fromarray(cropped_im)
        # print(type(tmp))
        cv2.imwrite(super_save_path, cropped_im)
    return None

if __name__ == '__main__':
    colorImg_path = r"/Users/feiyueyan/Desktop/xx_resized.jpeg"
    # 两步实现
    # super_res_img = call_super_resolution(colorImg_path)
    # cropped_super_img = resize_1800_to_1024(super_res_img)
    # 单步实现
    call_super_resolution_directly(colorImg_path)

    