# Notice that in this assignment you are not allowed to use torch optimizer and nn module.
import torch
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from math import log10

# Hyper Parameters 
input_size = 784
hidden_size = 128
num_epochs = 10
batch_size = 100
learning_rate = 1e-4  # 0.001  # adjust: e.g 1e-4

# MNIST Dataset 
train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),  
                            download=True)

test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


# initialize your parameters - randomly
N = batch_size
M = input_size
H = hidden_size
W1 = torch.rand(M, H)
W2 = torch.rand(H, M)


def non(X):
    return np.maximum(X, 0)


def non_prime(X):
    return torch.clamp(X, -1, 0) + 1


def calc_grad_W1(X, W1, W2):
    XW1 = torch.mm(X, W1)
    nonXW1 = non(XW1)
    nonXW1W2 = torch.mm(nonXW1, W2)
    derivX_hat = 2 * (nonXW1W2 - X)

    transpX = torch.transpose(X, 0, 1)
    transpW2 = torch.transpose(W2, 0, 1)
    nonprimeXW1 = non_prime(XW1)



    assert(XW1.size()[0] == N)
    assert(XW1.size()[1] == H)
    assert(nonXW1.size()[0] == N)
    assert(nonXW1.size()[1] == H)
    assert(nonXW1W2.size()[0] == N)
    assert(nonXW1W2.size()[1] == M)
    assert(derivX_hat.size()[0] == N)
    assert(derivX_hat.size()[1] == M)
    assert(transpX.size()[0] == M)
    assert(transpX.size()[1] == N)
    assert(transpW2.size()[0] == M)
    assert(transpW2.size()[1] == H)
    assert(nonprimeXW1.size()[0] == N)
    assert(nonprimeXW1.size()[1] == H)

    result = torch.mm(transpX, torch.mm(derivX_hat, transpW2) * nonprimeXW1)
    # assert dimension equal to W1
    assert(result.size()[0] == M)
    assert(result.size()[1] == H)
    return result


def calc_grad_W2(X, W1, W2):
    XW1 = torch.mm(X, W1)
    nonXW1 = non(XW1)
    nonXW1W2 = torch.mm(nonXW1, W2)
    derivX_hat = 2 * (nonXW1W2 - X)

    transp_nonXW1 = torch.transpose(nonXW1, 0, 1)

    assert(XW1.size()[0] == N)
    assert(XW1.size()[1] == H)
    assert(nonXW1.size()[0] == N)
    assert(nonXW1.size()[1] == H)
    assert(nonXW1W2.size()[0] == N)
    assert(nonXW1W2.size()[1] == M)
    assert(derivX_hat.size()[0] == N)
    assert(derivX_hat.size()[1] == M)
    assert(transp_nonXW1.size()[0] == H)
    assert(transp_nonXW1.size()[1] == N)

    result = torch.mm(transp_nonXW1, derivX_hat)
    # assert dimension equal to W2
    assert(result.size()[0] == H)
    assert(result.size()[1] == M)
    return result



# Train the Model
for epoch in range(num_epochs):
    for i, (images, _, ) in enumerate(train_loader):
        # Convert torch tensor to Variable
        images = images.view(-1, 28*28)
        targets = images.clone()
        X = targets  # for convenience with formulas
        
        # forward pass
        X_hat = torch.mm(non(torch.mm(X, W1)), W2)

        # loss calculation
        loss = torch.sum((X_hat - targets)**2)

        # gradient calculation and update parameters
        W1 = W1 - learning_rate * calc_grad_W1(X, W1, W2)
        W2 = W2 - learning_rate * calc_grad_W2(X, W1, W2)

        # check your loss 
        if (i+1) % 1 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss))

# Test the Model
avg_psnr = 0
for (images, _, ) in test_loader:
    images = images.view(-1, 28*28)
    targets = images.clone()
    # get your predictions
    predictions = torch.mm(non(torch.mm(targets, W1)), W2)
    
    # calculate PSNR
    mse = torch.mean((predictions - targets).pow(2))
    psnr = 10 * log10(1 / mse)
    avg_psnr += psnr
print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(test_loader)))
