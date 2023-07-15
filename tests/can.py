import sys
sys.path.append("..")

import torch
import numpy as np
from datasets.identity import MyDataset
from datasets import transform_

def test_model_cann_mse(model_path,datas,device):
    model_cann_mse=torch.load(model_path).eval()
    train_dataset = MyDataset(*datas[0:2])
    test_dataset = MyDataset(*datas[2:4])
    test_dataset_o = MyDataset(*datas[4:6])

    def test(idxs,s_idxs,isTrain=False,isTest=False):
      dataset = train_dataset if isTrain else test_dataset if isTest else test_dataset_o
      cols = len(s_idxs)
      
      origins=[]
      for i in range(cols):
          idx = idxs[s_idxs[i]]
          image,_=dataset[idx]
          o_image=image.numpy().reshape((92,92))
          origins.append(o_image)

      res=[]
      for i in range(cols):
          idx = idxs[s_idxs[i]]
          spec=dataset[idx][1].unsqueeze(0).to(device) 
          
          gen=model_cann_mse(spec).cpu().detach().numpy()
          o_gen=transform_(np.squeeze(np.squeeze(gen.astype("float32"))))
          o_gen=o_gen.numpy().reshape((92,92))
          res.append(o_gen)
      return origins,res

    return test
