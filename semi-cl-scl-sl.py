import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models     as models
import torch.nn.functional    as F
import numpy                  as np
import os
from networks import Our_ResNet
from loss import SupConLoss
from attacks import pgd_linf, pgd_linf_end2end
from torch.utils.data import Dataset
import argparse


def parse_option():
    parser = argparse.ArgumentParser('argument for training and test')
    parser.add_argument('--method', type=str, default='cl-scl',
                        choices=['cl-scl', 'cl-sl', 'scl-sl'], help='semi-supervised methods')
    parser.add_argument('--reload_encoder', type=bool, default= False, help='reloading the trained base encoder')
    parser.add_argument('--reload_classifier', type=bool, default= False, help='reloading the trained linear classifier')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--percentage', type=float, default=0.5,
                        help='percentage of labeled data')
    parser.add_argument('--numEpochs', type=int, default=200,
                        help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--projectionDim', type=int, default=100,help='projection dimension')
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    parser.add_argument('--learningRate', type=float, default=3e-4)
    parser.add_argument('--featuresDim', type=int, default=2048)
    parser.add_argument('--trial', type=int, default=0,help='id for recording runs')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'], help='dataset')
    parser.add_argument('--eps_t2', type=float, default=4/255, help='eps for adversarial:threat model II')
    parser.add_argument('--iter_t2', type=int, default=40, help='numer of iterations for generating adversarial:threat model II')
    parser.add_argument('--alpha', type=float, default=1e-2, help= 'movement multiplier per iteration in adversarial examples')

    opt = parser.parse_args()
    
    # set the path according to the environment    
    opt.save_path = './save/ST/{}_models'.format(opt.dataset)
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
 
 
class DatasetMaker(Dataset):
    def __init__(self, data,targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        

    def __len__(self):
        l = self.data.size(0)
        return l

    def __getitem__(self,index) :

        if torch.is_tensor(index):
            index=index.tolist()

        num  = self.data[0].shape[0]
        img  =   torch.permute(self.data[index],(2,0,1))
        
        
        class_label = self.targets[index]

        if self.transform:
            img1 = self.transform(img)
            img2 = self.transform(img)
        
        return ([img, img1, img2], class_label)
    
    

opt = parse_option()



def set_loader(opt):

    trainCLTransform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(size=32),
            torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
            torchvision.transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2)])

    trainSupTransform = transforms.Compose([
        transforms.Pad(5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32)])



    trainEvalTransform = transforms.Compose([
            transforms.ToTensor()])

    testTransform = transforms.Compose([
            transforms.ToTensor()])


    if opt.dataset == 'cifar10':
        opt.n_cls = 10
        trainCLDataset     = torchvision.datasets.CIFAR10(root='./data/' ,train=True,  download=True)
        trainEvalDataset   = torchvision.datasets.CIFAR10(root='./data/' ,train=True,  transform=trainEvalTransform,  download=True)
        testDataset        = torchvision.datasets.CIFAR10(root='./data/' ,train=False, transform=testTransform)

    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
        trainCLDataset     = torchvision.datasets.CIFAR100(root='./data/' ,train=True, download=True)
        trainEvalDataset   = torchvision.datasets.CIFAR100(root='./data/' ,train=True,  transform=trainEvalTransform,  download=True)
        testDataset        = torchvision.datasets.CIFAR100(root='./data/' ,train=False, transform=testTransform)

    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset)) 

    idx_labeld = torch.empty([0])
    idx_UNlabeld = torch.empty([0])
    Dataset_Labeled = []
    Dataset_UNLabeled = []
    Target_Labeled = []
    Target_UNLabeled = []

    for i in range(opt.n_classes):
        idx_class_i = np.asarray(trainCLDataset.targets) == i
        
        datas_i = trainCLDataset.data[idx_class_i]
        
        Dataset_Labeled.append( torch.tensor( datas_i[:int(opt.percentage*datas_i.shape[0]) ] )  )
        Target_Labeled.append(torch.tensor([i]*int(opt.percentage*datas_i.shape[0])) ) 
        
        Dataset_UNLabeled.append(torch.tensor( datas_i[int(opt.percentage*datas_i.shape[0]): ]   ) )
        Target_UNLabeled.append(torch.tensor([i]*(datas_i.shape[0]-int(opt.percentage*datas_i.shape[0])  )   )) 
    
    Dataset_Labeled=torch.cat(Dataset_Labeled)/255.
    Target_Labeled=torch.cat(Target_Labeled)   
    Dataset_UNLabeled=torch.cat(Dataset_UNLabeled)/255.     
    Target_UNLabeled=torch.cat(Target_UNLabeled)  

    bs_labeled = int(opt.batch_size*opt.percentage)
    bs_UNlabeled = int(opt.batch_size*(1-opt.percentage))

    if opt.method == 'cl-sl' or opt.method =='scl-sl':
        DataSet_Labeled = DatasetMaker(Dataset_Labeled,Target_Labeled,trainSupTransform)
        DataSet_UNLabeled = DatasetMaker(Dataset_UNLabeled,Target_UNLabeled,trainCLTransform)
    else: 
        #opt.method == 'cl-scl':
        DataSet_Labeled = DatasetMaker(Dataset_Labeled,Target_Labeled,trainCLTransform)
        DataSet_UNLabeled = DatasetMaker(Dataset_UNLabeled,Target_UNLabeled,trainCLTransform)

    trainloader_Labeled = torch.utils.data.DataLoader(dataset = DataSet_Labeled , shuffle=True, batch_size=bs_labeled, drop_last=True)
    trainloader_UNLabeled = torch.utils.data.DataLoader(dataset = DataSet_UNLabeled , shuffle=True, batch_size=bs_UNlabeled, drop_last=True)
  
        
    trainEvalLoader = torch.utils.data.DataLoader(dataset=trainEvalDataset, batch_size = opt.batch_size, num_workers= opt.num_workers, pin_memory=True, shuffle=True  , drop_last=True)
    testLoader      = torch.utils.data.DataLoader(dataset=testDataset,  batch_size = opt.batch_size, num_workers= opt.num_workers, pin_memory=True, shuffle=False, drop_last=True )
    
    trainloader_Labeled = torch.utils.data.DataLoader(dataset = DataSet_Labeled , shuffle=True, batch_size=bs_labeled, drop_last=True)
    trainloader_UNLabeled = torch.utils.data.DataLoader(dataset = DataSet_UNLabeled , shuffle=True, batch_size=bs_UNlabeled, drop_last=True)

    return trainloader_Labeled,trainloader_UNLabeled, trainEvalLoader,testLoader
        
    
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
    Linear2 = nn.Linear(opt.featuresDim,opt.n_classes)
    Linear2 = Linear2.to(device)
    EvalNet_SL  = EncoderWithHead(Encoder, Linear2)
    
    return CLNet,EvalNet,EvalNet_SL
    


class EncoderWithHead(nn.Module):
    def __init__(self, encoder, head):
        super(EncoderWithHead, self).__init__()
        self.encoder        = encoder
        self.head = head


    def forward(self, x):
        out = F.normalize(self.head(self.encoder(x)),dim=1)
        return out

 


def trainCLNet(opt,trainloader_Labeled,trainloader_UNLabeled,CLNet,criterion_CL,optimizer,device,EvalNet_SL = None,criterion_SL=None):
    totalStep = len(trainloader_Labeled)
    CLNet.encoder.train()
    CLNet.head.train()
    for epoch in range(opt.numEpochs):
        zip_loader = zip(trainloader_Labeled, trainloader_UNLabeled)
        for i, ((X1, labels1), (X2, labels2)) in enumerate(zip_loader):
            #######  Supervised
            if (opt.method == 'cl-sl' or opt.method == 'scl-sl'):
                 T1_x0_X1 = X1[1].to(device)
                 labels1 = labels1.to(device)
                 z1_T1_X1 = EvalNet_SL(T1_x0_X1)
                 loss1 = criterion_SL(z1_T1_X1, labels1).to(device)
            else:
                 loss1 = 0
             

            #######  Con
            if (opt.method == 'cl-sl' or opt.method == 'cl-scl'):
                 T1_x0_X2 = X2[1].to(device)
                 T2_x0_X2 = X2[2].to(device)
                 z1_T1_X2 = CLNet(T1_x0_X2)
                 z2_T2_X2 = CLNet(T2_x0_X2)
                 features1 = torch.cat([z1_T1_X2.unsqueeze(1), z2_T2_X2.unsqueeze(1)], dim=1)
                 loss2 = criterion_CL(features1).to(device)
            else:
                 loss2 = 0

            #######  SupCon
            if opt.method == 'scl-sl':
                 T1_x0_X2 = X2[1].to(device)
                 T2_x0_X2 = X2[2].to(device)
                 z1_T1_X2 = CLNet(T1_x0_X2)
                 z2_T2_X2 = CLNet(T2_x0_X2)
                 features2 = torch.cat([z1_T1_X2.unsqueeze(1), z2_T2_X2.unsqueeze(1)], dim=1)
                 loss3 = criterion_CL(features2,labels2).to(device)
            elif opt.method == 'cl-scl':
                 T1_x0_X1 = X1[1].to(device)
                 T2_x0_X1 = X1[2].to(device)
                 z1_T1_X1 = CLNet(T1_x0_X1)
                 z2_T2_X1 = CLNet(T2_x0_X1)
                 features2 = torch.cat([z1_T1_X1.unsqueeze(1), z2_T2_X1.unsqueeze(1)], dim=1)
                 loss3 = criterion_CL(features2,labels2).to(device)  
            else:
             loss3 = 0
            
            loss =  loss1 + loss2 + loss3

            
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
            # Forward pass
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
    totalStep = len(testLoader)
    EvalNet.encoder.eval()
    EvalNet.head.eval()
    total_acc_test = 0
    for i, (X, labels) in enumerate(testLoader):
            
            X = X.to(device)
            labels = labels.to(device)
            h = EvalNet.encoder(X)
            # Forward pass
            Z = EvalNet(X)  
            total_acc_test += (Z.max(dim=1)[1] == labels).sum().item()
    print('Acc_Test =', total_acc_test / len(testLoader.dataset),sep="\t")    
    return total_acc_test / len(testLoader.dataset)

    

def testEvalNet_adv(opt,testLoader,CLNet,EvalNet,criterion_adv,device):
    totalStep = len(testLoader)
    EvalNet.encoder.eval()
    CLNet.encoder.eval()
    EvalNet.head.eval()
    CLNet.head.eval()
    total_acc_test = 0
    for i, (X, labels) in enumerate(testLoader):
            
            X = X.to(device)
            delta = pgd_linf(CLNet, X, opt.eps_t1, opt.alpha, opt.iter_t1, criterion_adv, labels, opt.method, device)
            X_adv = (X + delta)
            labels = labels.to(device)
            # Forward pass
            Z2 = EvalNet(X_adv)
            total_acc_test += (Z2.max(dim=1)[1] == labels).sum().item()
           
    print('Acc_Test_under_Threat Model I =', total_acc_test / len(testLoader.dataset),sep="\t")
    return total_acc_test/len(testLoader.dataset)


    

def testEvalNet_adv_end2end(opt,testLoader,EvalNet,device):
    totalStep = len(testLoader)
    EvalNet.encoder.eval()
    EvalNet.head.eval()
    total_acc_test = 0
    
    for i, (X, labels) in enumerate(testLoader):
            
            X = X.to(device)
            labels = labels.to(device)
            delta = pgd_linf_end2end(EvalNet, X,labels, opt.eps_t2, opt.alpha, opt.iter_t2)
            X_adv = (X + delta)
            
            # Forward pass
            Z2 = EvalNet(X_adv)
            total_acc_test += (Z2.max(dim=1)[1] == labels).sum().item()
    print('Acc_Test_under_Threat Model II =', total_acc_test / len(testLoader.dataset),sep="\t")
    return total_acc_test/len(testLoader.dataset)


def main():
    opt = parse_option()
    trainloader_Labeled,trainloader_UNLabeled, trainEvalLoader,testLoader = set_loader(opt)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if (opt.method == 'cl-sl' or opt.method == 'scl-sl'):
        CLNet,EvalNet, EvalNet_SL =set_models(opt,device)
    else:
        CLNet,EvalNet, _ =set_models(opt,device)
    
    # Representation Learning Phase
    
    if opt.reload_encoder == True:
        PATH = opt.save_path+'/CLNet_'+opt.model_name+'.pt'
        CLNet.load_state_dict(torch.load(PATH))
    elif (opt.method == 'cl-sl' or opt.method == 'scl-sl'):
        criterion_CL = SupConLoss(temperature=opt.temp)
        criterion_SL = nn.CrossEntropyLoss()
        params = list(CLNet.parameters())+list(EvalNet_SL.parameters())
        optimizer = torch.optim.Adam(params, lr=opt.learningRate)
        trainCLNet(opt,trainloader_Labeled,trainloader_UNLabeled,CLNet,criterion_CL,optimizer,device,EvalNet_SL,criterion_SL)
    else: 
        criterion_CL = SupConLoss(temperature=opt.temp)
        optimizer = torch.optim.Adam(CLNet.parameters(), lr=opt.learningRate)
        trainCLNet(opt,trainloader_Labeled,trainloader_UNLabeled,CLNet,criterion_CL,optimizer,device)
    # Linear Classification Phase
    if opt.reload_classifier == True:
        PATH = opt.save_path+'/EvalNet_'+opt.model_name+'.pt'
        EvalNet.load_state_dict(torch.load(PATH))
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(EvalNet.head.parameters(), lr=opt.learningRate)
        trainEvalNet(opt,trainEvalLoader,EvalNet,criterion,optimizer,device)
        
    # Test on Clean data
    testEvalNet(opt,testLoader,EvalNet,device)
    
    
    # Test under Threat Model II
    testEvalNet_adv_end2end(opt,testLoader,EvalNet,device)
    
    
if __name__ == '__main__':
    main()







