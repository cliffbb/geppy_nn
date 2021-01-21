# imports
import numpy as np
from skimage import io
from skimage.transform import rotate
from skimage.util import img_as_ubyte
import os
from tqdm import tqdm
from fastai.vision.all import get_image_files
from torchvision.datasets.vision import VisionDataset


# ISPRS dataset mask codes
mask_codes = {0: (255, 255, 255),   # Impervious surfaces (white)
              1: (0, 0, 255),       # Buildings (blue)
              2: (0, 255, 255),     # Low vegetation (cyan)
              3: (0, 255, 0),       # Trees (green)
              4: (255, 255, 0),     # Cars (yellow)
              5: (255, 0, 0),       # Clutter (red)
              6 : (0, 0, 0)         # Undefined (black)
              }


def mask_to_img(mask, codes=None):
    """mask (2D) to rgb image (3D)"""
    if codes is None:
        codes = mask_codes
    img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i, c in codes.items():
        px = mask == i
        img[px] = c
    return img


def img_to_mask(img, codes=None):
    """rgb image (3D) to mask (2D)"""
    if codes is None:
        codes = mask_codes
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i, c in codes.items():
        px = np.all(img == np.array(c).reshape((1, 1, 3)), axis=2)
        mask[px] = i
    return mask


def extract_patches(img, stride=64, size=(128, 128)):
    """extract image patches across a large aerial image"""
    patches = []
    for x in range(0, img.shape[0], stride):
        for y in range(0, img.shape[1], stride):
            patch = img[x:x + size[0], y:y + size[1]]
            if patch.shape[:2] == size:
                patches.append(patch)
    return patches


def transforms_(patch, flip_v=False, flip_h=False, rotation=None):
    """data augmentation"""
    patches = [patch]
    if rotation == None:
        rotation = [90, 180, 270]
    for angle in rotation:
        patches.append(rotate(patch, angle))
    if flip_v:
        patches.append(np.flipud(patch))
    if flip_h:
        patches.append(np.fliplr(patch))
    return patches


def create_trainset(img_ids, img_files, msk_files, dst_img_folder, dst_msk_folder, patch_size=(128,128),
                    step_size=64, transform=False):
    image_files = str(img_files)
    mask_files = str(msk_files)

    if not os.path.isdir(str(dst_img_folder)):
        os.makedirs(str(dst_img_folder))
    else:
        raise Exception("Directory '{}' exists".format(str(dst_img_folder)))

    if not os.path.isdir(str(dst_msk_folder)):
        os.makedirs(str(dst_msk_folder))
    else:
        raise Exception("Directory '{}' exists".format(str(dst_msk_folder)))

    n_masks = 0
    n_imgs = 0
    for id in tqdm(img_ids):
        image_samples = []
        mask_samples = []
        image = io.imread(image_files.format(*id))
        mask = img_to_mask(io.imread(mask_files.format(*id)))
        # mask = io.imread(mask_files.format(*id))

        print('Extracting image patches from Tile {}...'.format(*id))
        for ip in extract_patches(image, step_size, patch_size):
            if transform: image_samples.extend(transforms_(ip))
            else: image_samples.append(ip)

        print('Extracting mask patches from Tile {}...'.format(*id))
        for mp in extract_patches(mask, step_size, patch_size):
            if transform:mask_samples.extend(transforms_(mp))
            else: mask_samples.append(mp)

        # if transform:mask_samples.extend(img_to_mask(np.asarray(transforms_(mp))))
            # else: mask_samples.append(img_to_mask(mp))

        print('Saving image patches of Tile {}...'.format(*id))
        for i, img in enumerate(image_samples):
            io.imsave('{}/{}.png'.format(str(dst_img_folder), i+n_imgs), img_as_ubyte(img))

        print('Saving mask patches of Tile {}...'.format(*id))
        for j, mask in enumerate(mask_samples):
            io.imsave('{}/{}.png'.format(str(dst_msk_folder), j+n_masks), img_as_ubyte(mask))

        n_imgs += len(mask_samples)
        n_masks += len(image_samples)
        del(mask_samples, image_samples)

    print('All done !\n The images have been saved in {}\n The masks have been saved in {}'.format(
            dst_img_folder, dst_msk_folder))
    print('The training set consists of {} images with {} masks'.format(n_imgs, n_masks))


class ISPRSDataset(VisionDataset):
    def __init__(self, img_dir, msk_dir, transform=None, target_transform=None, transforms=None):
        super(ISPRSDataset, self).__init__(transform, target_transform, transforms)
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.img_files = get_image_files(self.img_dir)
        self.get_mask = lambda x: self.msk_dir/f'{x.stem}{x.suffix}'

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        mask_file = self.get_mask(img_file)
        img = io.imread(img_file)
        mask = np.asarray(io.imread(mask_file))
        if self.transform is not None:
            img = self.transforms(img, mask)
        return img, mask


__all__ = ['mask_to_img', 'img_to_mask', 'create_trainset', 'ISPRSDataset']