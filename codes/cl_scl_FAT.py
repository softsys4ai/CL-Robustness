import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models     as models
import torch.nn               as nn
import torch.nn.functional    as F
import numpy                  as np
import os
from networks import Our_ResNet
from loss import SupConLoss
from attacks import pgd_linf_end2end
import argparse


def parse_option():
    parser = argparse.ArgumentParser('argument for training and test')
    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SimCLR', 'SupCon'], help='Contrastive learning methods')
    parser.add_argument('--scenario', type=str, default='AT-ّFull-AT',
                        choices=['AT-Full-AT', 'ST-Full-AT'], help='Two different scenario of training')
    parser.add_argument('--reload_model', type=bool, default= False, help='reloading the trained model')
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
    parser.add_argument('--eps_AT', type=float, default=(8/255), help='eps for Adversarial Training')
    parser.add_argument('--iter_AT', type=int, default=5, help='number of iterations for generating adversarial in Adversarial Training')
    parser.add_argument('--eps_t2', type=float, default=(4/255), help='eps for adversarial:threat model II')
    parser.add_argument('--iter_t2', type=int, default=40, help='numer of iterations for generating adversarial:threat model II')
    parser.add_argument('--alpha', type=float, default=1e-2, help='Movement multiplier per iteration in adversarial examples')
   

    opt = parser.parse_args()
    
    opt.model_name = '{}_{}_{}_bsz_{}_epoch_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.batch_size, opt.numEpochs, opt.trial)
        
        
    if opt.scenario == 'AT-Full-AT':
        opt.load_path = './save/AT-ST/{}_models'.format(opt.dataset)
        opt.save_path = './save/AT-Full_AT/{}_models'.format(opt.dataset)
    else:
        opt.load_path = './save/ST-ST/{}_models'.format(opt.dataset)
        opt.save_path = './save/ST-Full_AT/{}_models'.format(opt.dataset)
        
    if not os.path.isdir(opt.save_path):
        os.makedirs(opt.save_path)
        
    if opt.dataset == 'cifar10':
        opt.n_classes = 10
    elif opt.dataset == 'cifar100':
        opt.n_classes = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
        
    return opt
 

def set_loader(opt):
    trainEvalTransform = transforms.Compose([
            transforms.ToTensor()])

    testTransform = transforms.Compose([
            transforms.ToTensor()])
            
    if opt.dataset == 'cifar10':
        trainEvalDataset   = torchvision.datasets.CIFAR10(root='./data/' ,train=True,  transform=trainEvalTransform,  download=True)
        testDataset        = torchvision.datasets.CIFAR10(root='./data/' ,train=False, transform=testTransform)
    elif opt.dataset == 'cifar100':
        trainEvalDataset   = torchvision.datasets.CIFAR100(root='./data/' ,train=True,  transform=trainEvalTransform,  download=True)
        testDataset        = torchvision.datasets.CIFAR100(root='./data/' ,train=False, transform=testTransform)
    else:
            raise ValueError('dataset not supported: {}'.format(opt.dataset))
     
    trainEvalLoader = torch.utils.data.DataLoader(dataset=trainEvalDataset, batch_size= opt.batch_size, num_workers= opt.num_workers, pin_memory=True, shuffle=True  , drop_last=True)
    testLoader      = torch.utils.data.DataLoader(dataset=testDataset,  batch_size= opt.batch_size, num_workers= opt.num_workers, pin_memory=True, shuffle=False, drop_last=True )
             
    return trainEvalLoader,testLoader


def set_models(opt,device):
    
    ResNet = Our_ResNet()
    Encoder = ResNet.to(device)
    MLP = nn.Sequential( nn.Linear(opt.featuresDim, opt.featuresDim   ),
                         nn.ReLU(inplace=True),
                         nn.Linear(opt.featuresDim, opt.projectionDim ) )
    MLP = MLP.to(device)
    Linear = nn.Linear(opt.featuresDim, opt.n_classes)
    Linear = Linear.to(device)
    CLNet    = EncoderWithHead(Encoder, MLP)
    EvalNet  = EncoderWithHead(Encoder, Linear)
    
    return CLNet,EvalNet


class EncoderWithHead(nn.Module):
    def __init__(self, encoder, head):
        super(EncoderWithHead, self).__init__()
        self.encoder        = encoder
        self.head = head

    def forward(self, x):
        out = F.normalize(self.head(self.encoder(x)),dim=1)
        return out




def trainEvalNet_Ro(opt,trainEvalLoader,EvalNet,criterion,optimizer,device):
    totalStep = len(trainEvalLoader)
    EvalNet.encoder.train()
    EvalNet.head.train()
    for epoch in range(opt.numEpochs):
        for i, (X, labels) in enumerate(trainEvalLoader):
            X = X.to(device)
            labels = labels.to(device)
            delta = pgd_linf_end2end(EvalNet, X, labels, opt.eps_AT, opt.alpha, opt.iter_AT)
            X_adv = (X + delta)
            h = EvalNet.encoder(X)
            h_adv = EvalNet.encoder(X_adv)
            Z =  EvalNet.head(h)
            Z_adv =  EvalNet.head(h_adv)
            loss1 = criterion(Z, labels)
            loss2 = criterion(Z_adv, labels)
            loss = loss1 + loss2
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
    print('Acc_Test =', total_acc_test / len(testLoader.dataset),sep="\t")
    return total_acc_test / len(testLoader.dataset)
    



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
            Z2 = EvalNet(X_adv)
            predicted2 = Z2.argmax(1)
            total_acc_test += (Z2.max(dim=1)[1] == labels).sum().item()
    print('Acc_Test =', total_acc_test / len(testLoader.dataset),sep="\t")        
    return total_acc_test/len(testLoader.dataset)


def main():
    opt = parse_option()
    trainEvalLoader,testLoader = set_loader(opt)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    CLNet,EvalNet =set_models(opt,device)
    
    # Loading the Trained Model
    PATH = opt.load_path+'/CLNet_'+opt.model_name+'.pt'
    CLNet.load_state_dict(torch.load(PATH))
    

    if opt.reload_model == True:
        PATH = opt.save_path+'/EvalNet_'+opt.model_name+'.pt'
        EvalNet.load_state_dict(torch.load(PATH))
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(EvalNet.parameters(), lr=opt.learningRate)
        trainEvalNet_Ro(opt,trainEvalLoader,EvalNet,criterion,optimizer,device)
    
    
    # Test on Clean Data
    testEvalNet(opt,testLoader,EvalNet,device)

    
    # Test under Threat Model II
    testEvalNet_adv_end2end(opt,testLoader,EvalNet,device)
    
    
if __name__ == '__main__':
    main()

