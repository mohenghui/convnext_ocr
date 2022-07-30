from importlib.resources import path
from tkinter import Image, Variable
from sympy import im
import torch
from torchvision.transforms import transforms
import os
from dataset.dataset import FontData
from train import DEVICE, INPUT_SIZE
# class={}
transform_test=transforms.Compose([
    transforms.Resize((INPUT_SIZE,INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])
model=torch.load("model_12")
model.eval()
model.to(DEVICE)

test_path="data/test/"
testList=os.listdir(test_path)
for file in testList:
    img=Image.open(os.path.join(test_path,file))
    img=transform_test(img)
    img.unsqueeze_(0)
    img=Variable(img).to(DEVICE)
    out=model(img)

    _,pred=torch.max(out.data,1)
    print("Image Name:{},predice:{}".format(file,classs[pred.data.item()]))


dataset_test=FontData(test_path,transform_test,test=True)
print(len(dataset_test))

for index,(img,label) in enumerate(dataset_test):
    img.unsqueeze_(0)
    data=Variable(img).to(DEVICE)
    output=model(data)
    _,pred=torch.max(output.data,1)
    print('Image Name:{},predict:{}'.format(dataset_test.imgs[index], classes[pred.data.item()]))