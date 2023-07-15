import torch.nn as nn

#单一全连接 单一通道  存在前conv 无后conv
class Transmission_Matrix(nn.Module):
    def __init__(self,Win,Wout,act=nn.Sigmoid):  #batchsize * channelnum * W * W,  
        super(Transmission_Matrix, self).__init__()
        self.Win = Win
        self.Wout = Wout
        self.Matrix = nn.Sequential(
            nn.Linear(self.Win*self.Win, self.Wout*self.Wout, bias=True),
            act()
        )
        
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
            nn.BatchNorm2d(Temp_feature_nums),
            nn.Conv2d(Temp_feature_nums,Temp_feature_nums,3, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(Temp_feature_nums),
            nn.Conv2d(Temp_feature_nums,1,3, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(1),
        )
        self.Matrix_r = Transmission_Matrix(Win,Wout)
        
    def forward(self,input):
        result =self.Matrix_r(self.head(input))
        return result