from torch.utils.data import Dataset
from . import transform


class MyDataset(Dataset):
      def __init__(self,train_original, train_speckle):
          self.train_original=train_original.astype("float32")
          self.train_speckle=train_speckle.astype("float32")
          
      def __len__(self):
          return self.train_original.shape[0]
          
      def __getitem__(self,idx):
        return transform(self.train_original[idx,:,:,:]),transform(self.train_speckle[idx,:,:,:])
      