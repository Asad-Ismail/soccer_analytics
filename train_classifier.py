import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from utils.torch_utils import select_device, load_classifier, time_synchronized
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import numpy as np


def load_dataset(train_dir="",test_dir="",batch_size=5):
    train_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    test_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir,transform=test_transforms)
    train_loader = torch.utils.data.DataLoader(dataset = train_data,batch_size = batch_size,shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = train_data,batch_size = batch_size,shuffle = True)
    return train_loader,test_loader,train_data.classes


def test(net,test_loader,batch_size,classes):
    print('*'*20)
    print("Evaluating Network")
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images=images.to(device)
            labels=labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
        for i in range(len(classes)):
            print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

def training(net,train_loader,test_loader,device,classes,epochs=2,lr=0.001):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    for epoch in range(epochs):  # loop over the dataset multiple times
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs=inputs.to(device)
            labels=labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            loss = loss.item()
            if i % 50 == 49:    # print every 50 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, loss))
            #running_loss = 0.0
        test(net,test_loader,labels.shape[0],classes)
    PATH = './soccer_classify.pth'
    torch.save(net.state_dict(), PATH)
    print('Finished Training')



def predict_image(model,image,test_transform):
    image_tensor = test_transforms(image)
    image_tensor.unsqueeze_(0)
    image_tensor=image_tensor.to(device)
    output = model(image_tensor)
    index = output.data.cpu().numpy().argmax()
    return index

if __name__=="__main__":
    train=True
    device = select_device("0")
    modelc = load_classifier(name='resnet101', n=4)  # initialize
    train_loader,test_loader,classes=load_dataset("/media/asad/adas_cv_2/caffe/train_classifier","/media/asad/adas_cv_2/caffe/train_classifier")
    modelc.to(device)
    if train:
        training(modelc,train_loader,test_loader,device,classes,epochs=100)
    else:
        test_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        modelc.load_state_dict(torch.load("./soccer_classify.pth"))
        modelc.eval()
        image_pil = Image.open("/media/asad/adas_cv_2/caffe/patches/9000.png")
        cls_idx=predict_image(modelc,image_pil,test_transforms)