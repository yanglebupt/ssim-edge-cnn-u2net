import sys
sys.path.append("..")

import torch
from datasets.normalize_0_1 import MyDataset


def test_mode_u2net_ann(model_path,datas,device):
    model_u2net_ann=torch.load(model_path).eval()

    train_dataset = MyDataset(*datas[0:2])
    test_dataset = MyDataset(*datas[2:4])
    test_dataset_o = MyDataset(*datas[4:6])

    def test(idxs,s_idxs,isTrain=False,isTest=False):
      dataset = train_dataset if isTrain else test_dataset if isTest else test_dataset_o
      cols = len(s_idxs)
      res=[]
      for i in range(cols):
          idx = idxs[s_idxs[i]]
          spec=dataset[idx][1].unsqueeze(0).to(device)
          gen=model_u2net_ann(spec)[0]
          o_gen=gen.cpu().detach().numpy().reshape((92,92))
          res.append(o_gen)
      return res
    
    return test