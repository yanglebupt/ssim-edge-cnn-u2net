import sys
sys.path.append("..")

import torch
from datasets.normalize_0_1 import MyDataset
from datasets import transform_

def test_model_cnn_u2net(model_cann_path,model_u2net_path,datas,device):
    
    model_cann=torch.load(model_cann_path).eval()
    model_u2net=torch.load(model_u2net_path).eval()

    train_dataset = MyDataset(*datas[0:2])
    test_dataset = MyDataset(*datas[2:4])
    test_dataset_o = MyDataset(*datas[4:6])

    def test(idxs,s_idxs,isTrain=False,isTest=False):
        dataset = train_dataset if isTrain else test_dataset if isTest else test_dataset_o
        cann_res=[]
        u2net_res=[]
        cols = len(s_idxs)

        for i in range(cols):
            idx = idxs[s_idxs[i]]
            spec=dataset[idx][1].unsqueeze(0).to(device)
            
            gen2=model_cann(spec)
            o_gen2=gen2.cpu().detach().numpy().reshape((92,92))
            cann_res.append(o_gen2)
            
            nosie=transform_(o_gen2.reshape((92,92,1)))
            gen4=model_u2net(nosie.unsqueeze(0).to(device))[0]
            o_gen4=gen4.cpu().detach().numpy().reshape((92,92))
            u2net_res.append(o_gen4+nosie.numpy().reshape((92,92)))  
        return cann_res,u2net_res
    return test