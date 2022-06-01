import os
import matplotlib.pyplot as plt


train_image_path = 'data/train/images'
train_label_path = 'data/train/label'

def listdir(path):
    '''返回目录下所有的文件，去掉一些常见的特殊目录，并通过相同的字典序保证两个文件名序列一一对应
    path: 目录
    '''
    files=os.listdir(path)
    if ".DS_Store" in files:
        files.remove(".DS_Store")
    if "checkpoint" in files:
        files.remove("checkpoint")
    files.sort()
    return files


class My_Dataloader(Dataset):
    def __init__(self, image_path, label_path):
        super(My_Dataloader, self).__init__()
        
        self.images = listdir(image_path)
        self.labels = listdir(label_path)
    def __getitem__(self, index):
        image = plt.imread(os.path.join(train_image_path, self.images[index]))
        label = plt.imread(os.path.join(train_label_path, self.labels[index]))
        return image, label 

    def __len__(self):
        return len(self.images)

