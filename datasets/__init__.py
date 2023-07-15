import numpy as np
import h5py
import torchvision.transforms as transforms

speckle_size = 120
image_size = 92

transform_resize = lambda size=speckle_size: transforms.Resize(size)

transform_ = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

transform = transforms.Compose([
    transforms.ToTensor()
])

def load_dataset(file_location,what_to_load,start,end):
    hf = h5py.File( file_location , 'r')
    if end==-1:
        fl = hf[what_to_load][start:]
    else:
        fl = hf[what_to_load][start:end]
    Rarray = np.array(fl)
    hf.close()
    return Rarray

def getPair(file_location,elm,start,end,train=True):
    folder='Training' if train else 'Testing'
    # Original Images
    to_load = folder + '/Original_images/' + elm
    original = load_dataset(file_location, to_load,start,end)
    # Speckle Patterns
    to_load = folder + '/Speckle_images/' + elm
    speckle = load_dataset(file_location, to_load,start,end)
    return original,speckle

def createDataset(file_location):
    train_test_elm = "ImageNet"
    ### Train data
    original,speckle = getPair(file_location,train_test_elm,0,-1)
    speckle = speckle.reshape((speckle.shape[0],speckle_size,speckle_size,1))

    ### Validation data
    Who = ['horse', 'cat', 'parrot', 'punch']
    test_original_o = None
    test_speckle_o = None

    for i in range(len(Who)):
        start=0
        end=-1
        train=False
        elm=Who[i]
            
        ori,spe = getPair(file_location,elm,start,end,train)
        test_original_o=ori if test_original_o is None else np.concatenate([test_original_o,ori],axis=0)
        test_speckle_o=spe if test_speckle_o is None else np.concatenate([test_speckle_o,spe],axis=0)  
        
    test_speckle_o = test_speckle_o.reshape((test_speckle_o.shape[0],speckle_size,speckle_size,1))

    total = original.shape[0]
    train_rate = int(0.75 * total)
    idxs=np.arange(total)
    np.random.seed(10)
    np.random.shuffle(idxs)
    train_idxs = idxs[0:train_rate]

    train_original = original[train_idxs,:,:,:]
    train_speckle = speckle[train_idxs,:,:,:]

    test_original = np.delete(original,train_idxs,0)
    test_speckle = np.delete(speckle,train_idxs,0)

    return (train_original,
            train_speckle,
            test_original,
            test_speckle,
            test_original_o,
            test_speckle_o)
