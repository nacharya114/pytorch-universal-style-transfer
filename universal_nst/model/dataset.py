from torch.utils.data import Dataset
import natsort
import os
from PIL import Image

class UnsupervisedImageFolder(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        all_imgs = os.listdir(root)
        self.total_imgs = natsort.natsorted(all_imgs)
    
    def __len__(self):
        return len(self.total_imgs)
    
    def __getitem__(self, idx):
        img_loc = os.path.join(self.root, self.total_imgs[idx])
        img = Image.open(img_loc).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def __repr__(self):
        fmt_str = 'Unsupervised Image Dataset' + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
    