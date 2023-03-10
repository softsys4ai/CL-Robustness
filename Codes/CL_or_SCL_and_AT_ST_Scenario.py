import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models     as models
import torch.nn.functional    as F
import numpy                  as np
import os
from Networks import Our_ResNet
from Attacks import pgd_linf,pgd_linf_end2end
from Loss import SupConLoss

import argparse


def parse_option():
    parser = argparse.ArgumentParser('argument for training and test')
    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SimCLR', 'SupCon'], help='Contrastive learning methods')
    parser.add_argument('--Reload_Encoder', type=bool, default= False, help='Reloading the trained base encoder')
    parser.add_argument('--Reload_Classifier', type=bool, default= False, help='Reloading the trained linear classifier')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--numEpochs', type=int, default=200,
                        help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--projectionDim', type=int, default=100,help='projection dimension')
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    parser.add_argument('--learningRate', type=float, default=3e-4)
    parser.add_argument('--featuresDim', type=int, default=2048, help='ResNet50 output feature dimension')
    parser.add_argument('--trial', type=int, default=0,help='id for recording runs')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'], help='dataset')
    parser.add_argument('--eps_AT', type=float, default=(8/255), help='eps for adversarial training')
    parser.add_argument('--iter_AT', type=int, default=5, help='number of iterations for generating adversarial in adversarial training')
    parser.add_argument('--eps_t1', type=float, default=(4/255), help='eps for adversarial:threat model I')
    parser.add_argument('--iter_t1', type=int, default=40, help='number of iterations for generating adversarial:threat model I')
    parser.add_argument('--eps_t2', type=float, default=(4/255), help='eps for adversarial:threat model II')
    parser.add_argument('--iter_t2', type=int, default=40, help='numer of iterations for generating adversarial:threat model II')
    parser.add_argument('--alpha', type=float, default=1e-2, help='Movement multiplier per iteration in adversarial examples')
   

    opt = parser.parse_args()
    
    # set the path according to the environment
    opt.save_path = './save/AT-ST/{}_models'.format(opt.dataset)
    opt.model_name = '{}_{}_{}_bsz_{}_epoch_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.batch_size, opt.numEpochs, opt.trial)

    if not os.path.isdir(opt.save_path):
        os.makedirs(opt.save_path)
        
    if opt.dataset == 'cifar10':
        opt.n_classes = 10
    elif opt.dataset == 'cifar100':
        opt.n_classes = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt
 

class Image_and_TwoAugmentedTransform:
    def __init__(self, transform1,transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        return [self.transform1(x),self.transform2(x), self.transform2(x)]
        
     

def set_loader(opt):

    trainCLTransform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=32),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor()])


    trainEvalTransform = transforms.Compose([
            transforms.ToTensor()])

    testTransform = transforms.Compose([
            transforms.ToTensor()])
            
    if opt.dataset == 'cifar10':
        trainCLDataset     = torchvision.datasets.CIFAR10(root='./data/' ,train=True,  transform=Image_and_TwoAugmentedTransform(testTransform,trainCLTransform),  download=True)
        trainEvalDataset   = torchvision.datasets.CIFAR10(root='./data/' ,train=True,  transform=trainEvalTransform,  download=True)
        testDataset        = torchvision.datasets.CIFAR10(root='./data/' ,train=False, transform=testTransform)
    elif opt.dataset == 'cifar100':
        trainCLDataset     = torchvision.datasets.CIFAR100(root='./data/' ,train=True,  transform=Image_and_TwoAugmentedTransform(testTransform,trainCLTransform),  download=True)
        trainEvalDataset   = torchvision.datasets.CIFAR100(root='./data/' ,train=True,  transform=trainEvalTransform,  download=True)
        testDataset        = torchvision.datasets.CIFAR100(root='./data/' ,train=False, transform=testTransform)
    else:
            raise ValueError('dataset not supported: {}'.format(opt.dataset))   
    
    trainCLLoader   = torch.utils.data.DataLoader(dataset=trainCLDataset,   batch_size= opt.batch_size, num_workers= opt.num_workers, pin_memory=True, shuffle=True  , drop_last=True)
    trainEvalLoader = torch.utils.data.DataLoader(dataset=trainEvalDataset, batch_size= opt.batch_size, num_workers= opt.num_workers, pin_memory=True, shuffle=True  , drop_last=True)
    testLoader      = torch.utils.data.DataLoader(dataset=testDataset,  batch_size= opt.batch_size, num_workers= opt.num_workers, pin_memory=True, shuffle=False, drop_last=True )
    return trainCLLoader,trainEvalLoader,testLoader
    
def set_models(opt,device):
    
    ResNet = Our_ResNet()
    Encoder = ResNet.to(device)
    MLP = nn.Sequential( nn.Linear(opt.featuresDim, opt.featuresDim   ),
                         nn.ReLU(inplace=True),
                         nn.Linear(opt.featuresDim, opt.projectionDim ) )
    MLP = MLP.to(device)
    Linear = nn.Linear(opt.featuresDim,opt.n_classes)
    Linear = Linear.to(device)
    CLNet    = EncoderWithHead(Encoder, MLP)
    EvalNet  = EncoderWithHead(Encoder, Linear)
    return CLNet, EvalNet

class EncoderWithHead(nn.Module):
    def __init__(self, encoder, head):
        super(EncoderWithHead, self).__init__()
        self.encoder        = encoder
        self.head = head


    def forward(self, x):
        out = F.normalize(self.head(self.encoder(x)),dim=1)
        return out


    
    

def trainCLNet_Ro(opt,trainCLLoader,CLNet,criterion,optimizer,criterion_adv,device):
    totalStep = len(trainCLLoader)
    CLNet.encoder.train()
    CLNet.head.train()
    for epoch in range(opt.numEpochs):
        for i, (X, labels) in enumerate(trainCLLoader):
          x0 = X[0].to(device)
          x1 = X[1].to(device)
          x2 = X[2].to(device)
          delta1 = pgd_linf(CLNet, x0, opt.eps_AT, opt.alpha, opt.iter_AT, criterion_adv,labels,opt.method,device)
          X_adv1 = (x0 + delta1)
          # Forward pass
          z1_x0 = CLNet(x0)
          z2_x0 = CLNet(X_adv1)
          features1 = torch.cat([z1_x0.unsqueeze(1), z2_x0.unsqueeze(1)], dim=1)
          if opt.method == 'SupCon':
            loss1 = criterion(features1, labels).to(device)
          elif opt.method == 'SimCLR':
            loss1 = criterion(features1).to(device)
          z1_x1 = CLNet(x1)
          z2_x2 = CLNet(x2)
          features2 = torch.cat([z1_x1.unsqueeze(1), z2_x2.unsqueeze(1)], dim=1)
          if opt.method == 'SupCon':
            loss2 = criterion(features2, labels).to(device)
          elif opt.method == 'SimCLR':
            loss2 = criterion(features2).to(device)
          loss = loss1 + loss2
            # Backward and optimize
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          if (i+1) % 1 == 0:
            test_Accuracy = 0 #testAccuracy()
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch+1, opt.numEpochs, i+1, totalStep, loss.item()),flush=True)
    PATH = opt.save_path+'/CLNet_'+opt.model_name+'.pt'
    torch.save(CLNet.state_dict(), PATH)


def trainEvalNet(opt,trainEvalLoader,EvalNet,criterion,optimizer,device):
    totalStep = len(trainEvalLoader)
    EvalNet.encoder.eval()
    EvalNet.head.train()
    for epoch in range(opt.numEpochs):
        for i, (X, labels) in enumerate(trainEvalLoader):
            X = X.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                h = EvalNet.encoder(X)
            Z =  EvalNet.head(h)
            loss = criterion(Z, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 1 == 0:
                test_Accuracy = 0 #testAccuracy()
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch+1, opt.numEpochs, i+1, totalStep, loss.item()),flush=True)
    PATH = opt.save_path+'/EvalNet_'+opt.model_name+'.pt'
    torch.save(EvalNet.state_dict(), PATH)
    
 
def testEvalNet(opt,testLoader,EvalNet,device):
    EvalNet.encoder.eval()
    EvalNet.head.eval()
    total_acc_test = 0
    for i, (X, labels) in enumerate(testLoader):
            X = X.to(device)
            labels = labels.to(device)
            Z =  EvalNet(X)
            total_acc_test += (Z.max(dim=1)[1] == labels).sum().item()
    print('Acc_Test on clean data =', total_acc_test / len(testLoader.dataset),sep="\t")
    return total_acc_test / len(testLoader.dataset)




def testEvalNet_adv(opt,testLoader,CLNet,EvalNet,criterion_adv,device):
    totalStep = len(testLoader)
    EvalNet.encoder.eval()
    EvalNet.head.eval()
    CLNet.encoder.eval()
    CLNet.head.eval()
    total_acc_test = 0
    for i, (X, labels) in enumerate(testLoader):
            X = X.to(device)
            delta = pgd_linf(CLNet, X, opt.eps_t1, opt.alpha, opt.iter_t1, criterion_adv,labels,opt.method,device)
            X_adv = (X + delta)
            labels = labels.to(device)
            # Forward pass
            Z2 = EvalNet(X_adv)
            predicted2 = Z2.argmax(1)
            total_acc_test += (Z2.max(dim=1)[1] == labels).sum().item()
    print('Acc_Test Under Threat Model I =', total_acc_test / len(testLoader.dataset),sep="\t")        
    return total_acc_test/len(testLoader.dataset)

    
def testEvalNet_adv_end2end(opt,testLoader,EvalNet,device):
    totalStep = len(testLoader)
    EvalNet.encoder.eval()
    EvalNet.head.eval()
    total_acc_test = 0
    for i, (X, labels) in enumerate(testLoader):
            X = X.to(device)
            labels = labels.to(device)
            delta = pgd_linf_end2end(EvalNet, X, labels, opt.eps_t2, opt.alpha, opt.iter_t2)
            X_adv = (X + delta)
            # Forward pass
            Z2 = EvalNet(X_adv)
            total_acc_test += (Z2.max(dim=1)[1] == labels).sum().item()
    print('Acc_Test Under Threat Model II =', total_acc_test / len(testLoader.dataset),sep="\t")        
    return total_acc_test/len(testLoader.dataset)



def main():
    opt = parse_option()
    trainCLLoader,trainEvalLoader,testLoader = set_loader(opt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CLNet, EvalNet =set_models(opt,device)
    
    # Representation Learning Phase
    if opt.Reload_Encoder == True:
        PATH = opt.save_path+'/CLNet_'+opt.model_name+'.pt'
        CLNet.load_state_dict(torch.load(PATH))
    else:
        criterion = SupConLoss(temperature=opt.temp)
        optimizer = torch.optim.Adam(CLNet.parameters(), lr=opt.learningRate)
        criterion_adv = SupConLoss(temperature=opt.temp)
        trainCLNet_Ro(opt,trainCLLoader,CLNet,criterion,optimizer,criterion_adv,device)
       
        

    # Linear Classification Phase
    if opt.Reload_Classifier == True:
        PATH = opt.save_path+'/EvalNet_'+opt.model_name+'.pt'
        EvalNet.load_state_dict(torch.load(PATH))
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(EvalNet.head.parameters(), lr=opt.learningRate)
        trainEvalNet(opt,trainEvalLoader,EvalNet,criterion,optimizer,device)
    
    
    
    # Test on Clean Data
    testEvalNet(opt,testLoader,EvalNet,device)
    
    # Test under Threat Model I
    criterion_adv = SupConLoss(opt.temp)
    testEvalNet_adv(opt,testLoader,CLNet,EvalNet,criterion_adv,device)
    
    # Test under Threat Model II
    testEvalNet_adv_end2end(opt,testLoader,EvalNet,device)
    
    
if __name__ == '__main__':
    main()


