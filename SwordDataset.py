from torch.utils.data import Dataset
import numpy as np
from skimage import io

class SwordDataset(Dataset):

    def __init__(self, dataframe, filename_root, train=True, transform=None):
        self.dataframe = dataframe
        self.image_filename_root = filename_root
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # image = np.fromstring(self.dataframe.iloc[idx, -1], sep=' ').astype(np.float32).reshape(-1, IMG_SIZE)
        img_name =  f"{self.image_filename_root}{idx}.jpg"
        image = io.imread(img_name)
        if self.train:
            keypoints = self.dataframe.iloc[idx].values.astype(np.float32)
        else:
            keypoints = None
        # sample = {'image': image, 'keypoints': keypoints}
        sample = (image, keypoints)
        if self.transform:
            # sample['image'] = self.transform(sample['image'])
            sample = (self.transform(sample[0]), keypoints)
        return sample