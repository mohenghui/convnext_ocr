
from calendar import EPOCH
from cgi import test
from re import A
from statistics import mode
import torch.optim as optim
from pyexpat import model
from torchvision.transforms import transforms
from sklearn.utils import shuffle
import torch
import numpy as np
from torchtoolbox.transform import Cutout
from dataset.dataset import FontData
from torch.autograd import Variable
import torch.nn as nn
from torchtoolbox.tools import mixup_data, mixup_criterion
from models.convnext import convnext_tiny,convnext_base
modellr=1e-4
BATCH_SIZE=1024
EPOCHS=300
INPUT_SIZE=64
NUMCLASS=6864
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_path="data/train"
valid_path="data/valid"
transform=transforms.Compose([
    transforms.Resize((INPUT_SIZE,INPUT_SIZE)),
    Cutout(),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])
transform_test=transforms.Compose([
    transforms.Resize((INPUT_SIZE,INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

dataset_train=FontData(train_path,transforms=transform,train=True)
dataset_valid=FontData(valid_path,transforms=transform_test,train=False)

train_loader=torch.utils.data.DataLoader(dataset_train,batch_size=BATCH_SIZE,shuffle=True)
valid_loader=torch.utils.data.DataLoader(dataset_valid,batch_size=BATCH_SIZE,shuffle=False)

#损失函数
criterion=nn.CrossEntropyLoss()
model_ft=convnext_base(pretrained=True)
model_input=model_ft.downsample_layers[0][0]

model_ft.downsample_layers[0][0]=nn.Sequential(nn.Conv2d(1,3,1,1),
                                model_input)

# for k,v in model_input.named_modules():   
#     print(k,"-",v)
#修改最后一层,in_features得到该层的输入
num_ftrs=model_ft.head.in_features
model_ft.head=nn.Linear(num_ftrs,NUMCLASS)#修改成对应层数
model_ft.to(DEVICE)
print(model_ft)
optimizer=optim.Adam(model_ft.parameters(),lr=modellr)
#余弦退火
cosine_schedule=optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=20,eta_min=1e-9)#一次学习率周期的迭代次数,学习率会下降到多少

alpha=0.2
def train(model,device,train_loader,optimizer,epoch):
    model.train()#挂入gpu，调整参数
    sum_loss=0
    total_num=len(train_loader.dataset)#计算总数据量
    print(total_num,len(train_loader))
    for batch_idx,(data,target) in enumerate(train_loader):#序号，数据，标签
        target=np.array(target).astype(int)
        target=torch.from_numpy(target)
        data,target=data.to(device,non_blocking=True),target.to(device,non_blocking=True)#并行处理
        data,labels_a,labels_b,lam=mixup_data(data,target,alpha) #将两个字何在一起
        optimizer.zero_grad()
        output=model(data)
        loss=mixup_criterion(criterion,output,labels_a,labels_b,lam)
        loss.backward()#反向传播
        optimizer.step()
        print_loss=loss.data.item()
        sum_loss+=print_loss
        if (batch_idx+1)%10==0: #批次,处理了多少数据,一共多少数据,处理了多少百分之len(train_loader)指的是一共有多少个batch,损失值
            print('Train Epoch:{}[{}]/{}({:.0f}%)]\tLoss:{:.6f}'
            .format(epoch,(batch_idx+1)*len(data),len(train_loader.dataset),100.*(batch_idx+1)/len(train_loader),loss.item()))
    ave_loss=sum_loss/len(train_loader)#计算平均损失
    print("epoch:{},loss:{}".format(epoch,ave_loss))
ACC=0
#验证
def val(model,device,valid_loader,epoch):
    global ACC
    model.eval()#不做参数更新
    test_loss=0
    correct=0
    total_num=len(valid_loader.dataset)
    print(total_num,len(valid_loader))#总图片个数，需要迭代的次数
    with torch.no_grad():
        for data,target in valid_loader:
            data,target=Variable(data).to(device),Variable(target).to(device)
            output=model(data)
            loss=criterion(output,target)
            _,pred=torch.max(output.data,1)#类似于softmax
            correct+=torch.sum(pred==target)#计算正确个数
            print_loss=loss.data.item()
            test_loss+=print_loss #累计损失
        correct=correct.data.item()
        acc=correct/total_num #计算出正确率
        avgloss=test_loss/len(valid_loader)
        print('\nVal set: Average loss:{:.4f},Accuracy: {}/{} ({:.0f}%)\n'.format
        (avgloss,correct,len(valid_loader.dataset),100*acc))
        if acc>ACC: #如果正确率比之前的正确率还大就保存模型
            torch.save(model_ft,"model_"+str(epoch)+"_"+str(round(acc,3))+'.pth')
            ACC=acc

for epoch in range(1,EPOCHS+1):
    train(model_ft,DEVICE,train_loader,optimizer,epoch)
    cosine_schedule.step()
    val(model_ft,DEVICE,valid_loader)
    