import torch
import torch.nn  as nn

def pgd_linf(model, X, epsilon, alpha, num_iter,criterion,labels, method,device):
    
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        z = model(X)
        z_adv = model(X+delta)
        features = torch.cat([z.unsqueeze(1), z_adv.unsqueeze(1)], dim=1)
        if method == 'SupCon':
            loss = criterion(features, labels).to(device)
        elif method == 'SimCLR':
            loss = criterion(features).to(device)
        loss.backward()
        with torch.no_grad():
            delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.data = torch.clamp(X + delta.data, min=0, max=1) - X
        delta.grad.zero_()
    return delta.detach()
	
	
def pgd_linf_end2end(model, X, labels, epsilon, alpha, num_iter):
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), labels)
        loss.backward()
        with torch.no_grad():
            delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.data = torch.clamp(X + delta.data, min=0, max=1) - X
        delta.grad.zero_()
    return delta.detach()
    


