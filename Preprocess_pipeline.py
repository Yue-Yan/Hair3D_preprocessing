# 这个代码将会汇总全部的预处理步骤
# 1. 生成2D colorImg对应的maskImg；
# 2. 对colorImg和maskImg进行4倍高清处理；【待研究】
# 3. 根据maskImg来抠取colorImg里面的头发部分；
# 4. 先实现1和3，后续测试2；


import cv2
import os
import argparse
import numpy as np
import PIL.Image
from keras.models import load_model
from keras.utils import CustomObjectScope
from scipy.misc import imread, imresize, imsave
from custom_objects import custom_objects
from skimage import img_as_ubyte # 参看第59行，为了不显示那个float32变uint8的警告
from tqdm import tqdm


class DetectHairGenerateMask(object):
    def __init__(self, model_dir, colorImg_dir, input_shape=[256, 256]):
        self.colorImg_dir = colorImg_dir
        self.model_dir = model_dir
        self.input_shape = input_shape

    def blend_img_with_mask(self, img, alpha, img_shape):
        mask = alpha >= 0.99
        mask_3 = np.zeros(img_shape, dtype='float32')
        mask_3[:,:,0] = 255
        mask_3[:,:,0] *= alpha
        result = img*0.5 + mask_3*0.5
        return np.clip(np.uint8(result), 0, 255)

    def generate_mask(self):
        with CustomObjectScope(custom_objects()):
            model = load_model(self.model_dir)
            model.summary()
        
        imgs = [f for f in os.listdir(self.colorImg_dir) if not f.startswith('.')] # .DS_STORE
        for img_x in imgs: # img_x: img_original 
            # img = imread(os.path.join(self.colorImg_dir, img_x), mode='RGB') # 原文，用下面的替换了
            import imageio
            img = imageio.imread(os.path.join(self.colorImg_dir, img_x), pilmode='RGB')
            img_shape = img.shape
            input_data = img.astype('float32')
            # input_data = imresize(img, self.input_shape) # 原文，用下面一句替换了
            input_data = np.array(PIL.Image.fromarray(img).resize(self.input_shape))

            input_data = input_data / 255.
            input_data = (input_data - input_data.mean()) / input_data.std()
            input_data = np.expand_dims(input_data, axis=0)
            output = model.predict(input_data)
            mask = cv2.resize(output[0,:,:,0], (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
            mask_name = img_x.split('.')[0] + '_mask.png'

            saved_mask_path = os.path.join(self.colorImg_dir, mask_name)
            # imsave(saved_mask_path, mask) # 原文，用下面的替换了
            # imageio.imwrite(saved_mask_path, mask) # 这么写，一直会警告，参见https://github.com/zhixuhao/unet/issues/125
            imageio.imwrite(saved_mask_path, img_as_ubyte(mask))

            # 暂时不用
            # img_with_mask = self.blend_img_with_mask(img, mask, img_shape)
            # blend_name = img_x.split('.')[0] + '_blend.jpg'
            # saved_blend_path = os.path.join(self.colorImg_dir, blend_name)
            # imsave(saved_blend_path, blend_name)
            # imsave(f'./imgs/feiyue_test/results/{img_x}', img_with_mask)
        return None


# color2D + reversed_maskImg = pair
class CutoutHair(object):
    def __init__(self, colorImg_dir):
        self.colorImg_dir = colorImg_dir

    def generate_materials(self, condition='original'):
        remove_Mac = [f for f in os.listdir(self.colorImg_dir) if not f.startswith('.')] # .DS_Store
        if condition == 'original':
            material = sorted([os.path.join(self.colorImg_dir, f) for f in remove_Mac if 'mask' not in f])
        elif condition == 'mask':
            material = sorted([os.path.join(self.colorImg_dir, f) for f in remove_Mac if 'mask' in f])
        elif condition == 'reversed':
            material = sorted([os.path.join(self.colorImg_dir, f) for f in remove_Mac if 'reversed' in f])
        else:
            return
        return material

    # zip colorImg and reversed_maskImg
    def zip_pairs(self):
        colorImg_list = self.generate_materials(condition='original')
        maskImg_list = self.generate_materials(condition='mask')
        crm_pairs = zip(colorImg_list, maskImg_list)
        return crm_pairs

    # maskImg --> reversed_maskImg
    def reverse_mask_value(self, maskImg_path): # 0 <--> 255
        maskImg = cv2.imread(maskImg_path) # maskImg_path: maskImg original
        maskImg_cp = maskImg.copy()
        tmp_black = np.where(maskImg_cp==0)
        tmp_white = np.where(maskImg_cp==255)
        maskImg_cp[tmp_black] = 255
        maskImg_cp[tmp_white] = 0
        reversed_maskImg_name = os.path.basename(maskImg_path).split('.')[0] + '_reversed.png'
        cv2.imwrite(self.colorImg_dir, reversed_maskImg_name)
        return maskImg_cp

    # reversed_maskImg + colorImg == cutout_hair
    def cutout_hair(self, colorImg_path, maskImg_path):
        colorImg = cv2.imread(colorImg_path)
        reversed_maskImg = self.reverse_mask_value(maskImg_path)
        cutoutImg = cv2.add(colorImg, reversed_maskImg)

        corresponding_name = os.path.basename(colorImg_path).split('.')[0] + '_cutout.png'
        cutoutImg_save_path = os.path.join(self.colorImg_dir, corresponding_name)
        cv2.imwrite(cutoutImg_save_path, cutoutImg)

    
if __name__ == '__main__':
    model_dir = "./models/CelebA_PrismaNet_256_hair_seg_model.h5"
    colorImg_dir = r"./colorImg_samples"
    try:
        dhgm = DetectHairGenerateMask(model_dir, colorImg_dir)
        dhgm.generate_mask()
        #---------------------------------------
        cut = CutoutHair(colorImg_dir)
        for colorImg_org, maskImg_org in cut.zip_pairs():
            print(f'{colorImg_org, maskImg_org}')
            cut.cutout_hair(colorImg_org, maskImg_org)
    except Exception as e:
        pass
