import cv2
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

    
class HDRDataset(Dataset):


    def __init__(self, data_dir, mode='train', transform=None):
        total_len = len(os.listdir(data_dir))//2
        self.data_dir = data_dir
        if mode == 'train':
            self.data_len = 3000
      
        elif mode == 'test':
            self.data_len = total_len - 3000

        self.transform = transform


    def __len__(self):
        return self.data_len


    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, '{}.jpg'.format(str(idx).zfill(4)))
        label_path = os.path.join(self.data_dir, '{}_gt.jpg'.format(str(idx).zfill(4)))
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        h1, w1 = np.shape(image)[:2]
        h2, w2 = np.shape(label)[:2]

        if h1 > h2:
            dh = (h1-h2)//2
            image = image[dh: dh+h2, :, :]
        elif h1 < h2:
            dh = (h2-h1)//2
            label = label[dh: dh+h1, :, :]
            
        if w1 > w2:
            dw = (w1-w2)//2
            image = image[:, dw: dw+w2, :]
        elif w1 < w2:
            dw = (w1-w2)//2
            label = label[:, dw: dw+w1, :]
        
        try:
            assert np.shape(image) == np.shape(label)
        except:
            print(np.shape(image), np.shape(label))
        
        
        if self.transform is not None:
            image, label = self.transform(image, label)
            
        return image, label



class TrainTransform():
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        
        
    def __call__(self, image, label):
        
        new_h, new_w = self.output_size

        h, w = np.shape(image)[:2]
        offset_h = np.random.randint(0, h - new_h)
        offset_w = np.random.randint(0, w - new_w)

        image = image[offset_h: offset_h + new_h, 
                                offset_w: offset_w + new_w]
        label = label[offset_h: offset_h + new_h, 
                                offset_w: offset_w + new_w]
        
        flip_prop = np.random.randint(0, 100)
        if flip_prop > 50:
            image = cv2.flip(image, 1)   
            label = cv2.flip(label, 1)   
        
        image = image.astype(np.float32)/255.0
        label = label.astype(np.float32)/255.0

        
        return image, label

    
    
class TestTransform():
    
    def __init__(self):
        pass
        
        
    def __call__(self, image, label):
        
        image = image.astype(np.float32)/255.0
        #label = label.astype(np.float32)/127.5 - 1
        
        return image, label



def get_train_loader(image_size, batch_size, data_dir):
    transform = TrainTransform(image_size)
    dataset = HDRDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=8)
    return dataloader



def get_test_loader(data_dir):
    transform = TestTransform()
    dataset = HDRDataset(data_dir, mode='test', transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, 
                            shuffle=False, num_workers=0)
    return dataloader



if __name__ == '__main__':
    data_dir = '/media/wangxinrui/新加卷/hdr+burst/hdr_burst'
    dataloader = get_train_loader(512, 8, data_dir) 
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        print(np.shape(batch[0]), np.shape(batch[1]))
        #pass

     
        
     

    


