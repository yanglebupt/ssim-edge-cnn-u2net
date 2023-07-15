import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
import matplotlib.pyplot as plt
import cv2
from scipy.stats import pearsonr
import math
from tqdm import tqdm

fontdict = {'fontsize' : 170}


def EME(img, L=8):
    m, n = img.shape
    number_m = math.floor(m/L)
    number_n = math.floor(n/L)
    m1 = 0
    E = 0
    for _ in range(number_m):
        n1 = 0
        for __ in range(number_n):
            A1 = img[m1:m1+L, n1:n1+L]
            rbg_min = np.amin(np.amin(A1))
            rbg_max = np.amax(np.amax(A1))
 
            if rbg_min > 0 :
                rbg_ratio = rbg_max/rbg_min
            else :
                rbg_ratio = rbg_max  ###
            E = E + np.log(rbg_ratio + 1e-5)
 
            n1 = n1 + L
        m1 = m1 + L
    E_sum = 20*E/(number_m*number_n)
    return E_sum


def normalize(img):
    return cv2.normalize(img.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)

def showImg(img,axis="off"):
    plt.imshow(normalize(img),cmap="gray",norm=plt.Normalize(0,1))
    plt.axis(axis)
        
def compareRes(img,gen):
    img=normalize(img)
    gen=normalize(gen)
    return {
        "mse":compare_mse(img,gen),
        "psnr":compare_psnr(img,gen,data_range=1), 
        "ssim":compare_ssim(img,gen,data_range=1),
        "pncc":pearsonr(img.flatten(), gen.flatten()),
        "emes":(EME(img),EME(gen))    
    }  

def fft(img):
    return np.fft.fft2(img)

def showfft(f_img,axis="off"):
    plt.imshow(np.log(1+ np.abs(np.fft.fftshift(f_img))),cmap='gray')
    plt.axis(axis)
    
def ifft(f_img):
    return np.abs(np.fft.ifft2(f_img))

def showCompare(models_train_res,filename,h_pad=10,w_pad=-10):
  rows = len(models_train_res)
  origin_train_list=models_train_res.pop(0)
  cols = len(origin_train_list)

  plt.figure(figsize=(cols*30,rows*30))
  for i in range(cols):
      pos=i+1
      plt.subplot(rows,cols,pos)
      o_image=origin_train_list[i]
      showImg(o_image)
      
      for j in range(len(models_train_res)):
          plt.subplot(rows,cols,pos+(j+1)*cols)
          o_gen=models_train_res[j][i]
          showImg(o_gen)
          res=compareRes(o_gen,o_image)  
          plt.title(res["ssim"],fontdict=fontdict)
  plt.tight_layout(h_pad=h_pad,w_pad=w_pad)
  plt.savefig(filename)
    
    
def predict_one(model,dataset,idx):
    image,nosie=dataset[idx]
    
    o_image=image.numpy().reshape((92,92))
    
    o_nosie=nosie.numpy().reshape((120,120))
    
    gen=model(nosie.unsqueeze(0).to(device))
    o_gen=gen.cpu().detach().numpy().reshape((92,92))
    
    res=compareRes(o_gen,o_image)
    return res

def calcPre(diff,th):
    # 大小大
    return len(np.where(diff>th)[0])/len(diff),len(np.where(diff<=0)[0])/len(diff),np.mean(diff)

def calc(val):
    # 大小大
    return np.min(val),np.max(val),np.mean(val),np.std(val)

def calcModel(model,train_dataset,test_dataset,test_dataset_o):
    train_res=[]
    train_nums = len(train_dataset)
    for i in tqdm(range(train_nums)):
        res=predict_one(model,train_dataset,i)
        train_res.append(res)
    
    test_res=[]
    test_nums = len(test_dataset)
    for i in tqdm(range(test_nums)):
        res=predict_one(model,test_dataset,i)
        test_res.append(res)
        
    val_res=[]
    val_nums = len(test_dataset_o)
    for i in tqdm(range(val_nums)):
        res=predict_one(model,test_dataset_o,i)
        val_res.append(res)

    return [train_res,test_res,val_res]