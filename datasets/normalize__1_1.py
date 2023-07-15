from torch.utils.data import Dataset
from . import transform_,transform_resize
import torchvision.transforms as transforms


class MyDataset(Dataset):
      def __init__(self,train_original, train_speckle, resize=None):
          self.train_original=train_original.astype("float32")
          self.train_speckle=train_speckle.astype("float32")
          self.transform_2 =  transforms.Compose([
              transform_.transforms[0],
              transform_resize(),
              transform_.transforms[1]
          ]) if resize else transform_
        
          
      def __len__(self):
          return self.train_original.shape[0]
          
      def __getitem__(self,idx):
        return self.transform_2(self.train_original[idx,:,:,:]), transform_(self.train_speckle[idx,:,:,:]/255)
      