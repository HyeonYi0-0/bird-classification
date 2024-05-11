import numpy as np
import cv2

from torch.utils.data import Dataset
from zipfile import ZipFile
from PIL import Image
from tqdm import tqdm

class ZipDataset(Dataset):
    def __init__(self, img_path_list, label_list, zipfile_path, transform=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transform = transform
        self.data = []

        # read zip file to save storage
        with ZipFile(zipfile_path, 'r') as zipObj:
            for img_path in tqdm(self.img_path_list):
                buf = zipObj.read(img_path[2:])
                image = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)

                pil_image = Image.fromarray(np.uint8(image))

                self.data.append(pil_image)


    def __getitem__(self, index):
        image = self.data[index]

        if self.transform is not None:
            image = self.transform(image)

        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.img_path_list)