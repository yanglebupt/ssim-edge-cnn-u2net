
def train_and_test(model,dataloader,device,loss_fn,optimizer=None,isTrain=True,isResidual=False):
    error=0
    for i,(original,speckle) in enumerate(dataloader):
        if isTrain:
            optimizer.zero_grad()
        
        original=original.to(device)
        speckle=speckle.to(device)

        gen=model(speckle)
        loss=loss_fn(gen,original-speckle if isResidual else original)
        
        if isTrain:
            loss.backward()
            optimizer.step()
            
        error+=loss.item()
        
    return error/(i+1)
