import torch.nn as nn

class Transmission_Matrix(nn.Module):
    def __init__(self,Win,Wout):   
        super(Transmission_Matrix, self).__init__()
        self.Win = Win
        self.Wout = Wout
        self.Matrix = nn.Linear(14400, self.Wout*self.Wout, bias=True)
        
    def forward(self, input):
        input = input.view(input.shape[0],input.shape[1],-1)
        out = self.Matrix(input)
        out = out.view(input.shape[0],input.shape[1],self.Wout,self.Wout)
        return out
    
class EnhancedNet(nn.Module):
    def __init__(self,Win,Wout,Temp_feature_nums):
        super(EnhancedNet, self).__init__()
        self.head=nn.Sequential(
            nn.Conv2d(1,Temp_feature_nums,3,padding=1),
            nn.Tanh(),
            nn.Conv2d(Temp_feature_nums,Temp_feature_nums,3, padding=1),
            nn.Tanh(),
            nn.Conv2d(Temp_feature_nums,1,3, padding=1),
            nn.Tanh()
        )
        self.Matrix_r = Transmission_Matrix(Win,Wout)
        self.tailAct=nn.Sigmoid()
        
    def forward(self,input):

        result = self.tailAct(self.Matrix_r(self.head(input)))
        return result