from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import csv
import math
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import numpy as np
from random import random, choice, shuffle
import cv2
from scipy.ndimage.filters import gaussian_filter
from io import BytesIO


# ============== Dataset ==============
class ImageDataset(Dataset):
    """
    CSV format: two columns -> path, label
    """

    def __init__(self, csv_path: str, is_train=False, opt=None):
        self.items = []
        self.is_train = is_train
        self.opt = opt

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert "path" in reader.fieldnames and "label" in reader.fieldnames, \
                f"{csv_path} must contain columns: path,label"

            for r in reader:
                path = r["path"]
                label = int(r["label"])
                self.items.append((path, label))

        if is_train:
            self.preprocess = transforms.Compose([
                transforms.Lambda(lambda img: translate_duplicate(img, 224)),
                transforms.Lambda(lambda img: data_augment(img, opt)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711]),
            ])
        else:
            self.preprocess = transforms.Compose([
                transforms.Lambda(lambda img: translate_duplicate(img, 224)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711]),
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        rgb = Image.open(path).convert("RGB")

        rgb = self.preprocess(rgb)
        return rgb, label


class ImageFolderWithPaths(ImageFolder):
    def __init__(self, root, transform=None, jpeg_quality=None, gaussian_sigma=None):
        super().__init__(root)
        self.original_samples = self.samples
        self.transform = transform
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma

    def __getitem__(self, item):
        path, target = self.samples[item]

        target = 0 if '0_real' in path else 1
        img = Image.open(path).convert("RGB")

        if self.gaussian_sigma is not None and np.random.random() < 0.1:
            img = gaussian_blur(img, self.gaussian_sigma)
        if self.jpeg_quality is not None and np.random.random() < 0.1:
            img = png2jpg(img, self.jpeg_quality)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = self._process_rgbd(img)

        return img, target

    def __len__(self):
        return len(self.samples)

    def _process_rgbd(self, rgb: Image.Image):
        rgb = translate_duplicate(rgb, 224)

        rgb = TF.center_crop(rgb, 224)

        rgb = TF.to_tensor(rgb)
        rgb = TF.normalize(rgb,
                           mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711])
        return rgb


def translate_duplicate(img, cropSize):
    if min(img.size) < cropSize:
        width, height = img.size

        new_width = width * math.ceil(cropSize / width)
        new_height = height * math.ceil(cropSize / height)

        new_img = Image.new(img.mode, (new_width, new_height))
        for i in range(0, new_width, width):
            for j in range(0, new_height, height):
                new_img.paste(img, (i, j))
        return new_img
    else:
        return img


def data_augment(img, opt):
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)

    return Image.fromarray(img)


def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality)  # ranging from 0-95, 75 is default
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}


def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)
