import time
import torch
import torch.nn as nn
import torch.utils as utils
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt


# Set Hyperparameters
# noise_type = 'gaussian_add'  # Possible values: 'gaussian_add', 'noise_salt_pepper', 'noise_masking' or None
noise_types = ['gaussian_add', 'noise_salt_pepper', 'noise_masking', 'None']
# finetune = False # see below..
num_epochs_autoencoder = 2  # 10
num_epochs_classifier = 7  # 30
batch_size = 128
learning_rate = 0.001
LAYER_DIMS = [16, 8, 8]

# Check if we can use CUDA
cuda_available = torch.cuda.is_available()

# Define image transformations & Initialize datasets
mnist_transforms = transforms.Compose([transforms.ToTensor()])
mnist_train = dset.MNIST('./data', train=True, transform=mnist_transforms, download=True)
mnist_test = dset.MNIST('./data', train=False, transform=mnist_transforms, download=True)

# For reproducibility
torch.manual_seed(123)
np.random.seed(123)

# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=2,
                                           drop_last=True)
testloader = torch.utils.data.DataLoader(dataset=mnist_test,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=2,
                                         drop_last=True)

# # Choose 5000 examples for transfer learning
# mask = np.random.randint(0, 60000, 5000)
# finetune_loader = torch.utils.data.DataLoader(dataset=mnist_train,
#                                               batch_size=batch_size,
#                                               shuffle=False,
#                                               sampler=SubsetRandomSampler(np.where(mask)[0]),
#                                               num_workers=2)


# Create Encoder and Decoder that subclasses nn.Module
class Encoder(nn.Module):
    """Convnet Encoder"""

    def __init__(self):
        super(Encoder, self).__init__()
        # 28 x 28 -> 14 x 14
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=LAYER_DIMS[0], kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=LAYER_DIMS[0]),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        # 14 x 14 -> 7 x 7
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=LAYER_DIMS[0], out_channels=LAYER_DIMS[1], kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=LAYER_DIMS[1]),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        # 7 x 7 -> 4 x 4
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=LAYER_DIMS[1], out_channels=LAYER_DIMS[2], kernel_size=(3, 3), padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=LAYER_DIMS[2]),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


class Decoder(nn.Module):
    """Convnet Decoder"""

    def __init__(self):
        super(Decoder, self).__init__()
        # 4 x 4 -> 7 x 7
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LAYER_DIMS[2], out_channels=LAYER_DIMS[1],
                               kernel_size=(3, 3), stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(LAYER_DIMS[1]),
        )
        # 7 x 7 -> 14 x 14
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LAYER_DIMS[1], out_channels=LAYER_DIMS[0],
                               kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(LAYER_DIMS[0]),
        )
        # 14 x 14 -> 28 x 28
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LAYER_DIMS[0], out_channels=1,
                               kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


# Create a Classifer for the Encoder features
class Classifier(nn.Module):
    """Convnet Classifier"""

    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=LAYER_DIMS[2], out_channels=10, kernel_size=(4, 4), padding=0),
        )

    def forward(self, x):
        out = self.classifier(x).squeeze()
        return out


def noise_additive_gaussian(imgs, sigma=.5):
    """
    Adds additive gaussian noise to images for the training of a DAE

    Args:
        imgs: A batch of images
        sigma: Standard deviation of the gaussian noise

    Returns:
        imgs_n: The noisy images

    """
    #######################################################################
    #                                                                     #
    # Apply additive Gaussian noise to the images                         #
    #                                                                     #
    #######################################################################

    # src: https://discuss.pytorch.org/t/writing-a-simple-gaussian-noise-layer-in-pytorch/4694

    mean = 0.0
    noise = Variable(imgs.data.new(imgs.size()).normal_(mean, sigma))
    imgs_n = imgs + noise

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return imgs_n


def noise_salt_pepper(imgs, noise_rate=0.5):
    """
    Adds salt&pepper noise to images for the training of a DAE

    Args:
        imgs: A batch of images
        noise_rate: Controls the amount of noise (higher=more noise)

    Returns:
        imgs_n: The noisy images

    """
    #######################################################################
    #                                                                     #
    # Apply Salt&Pepper noise to the images                               #
    #                                                                     #
    #######################################################################

    imgs_clone = imgs.clone().view(-1, 1)
    num_feature = imgs_clone.size(0)
    mn = imgs_clone.min()
    mx = imgs_clone.max()
    indices = np.random.randint(0, num_feature, int(num_feature * noise_rate))
    for elem in indices:
        if np.random.random() < 0.5:
            imgs_clone[elem] = mn
        else:
            imgs_clone[elem] = mx
    imgs_n = imgs_clone.view(imgs.size())

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return imgs_n


def noise_masking(imgs, drop_rate=0.5, tile_size=7):
    """
    Randomly sets tiles of images to zero for the training of a DAE

    Args:
        imgs: A batch of images
        drop_rate: Controls the amount of tile dropping (higher=more noise)
        tile_size: The size of the tiles to be dropped in pixels

    Returns:
        imgs_n: The noisy images

    """
    #######################################################################
    #                                                                     #
    # Apply masking to the images                                         #
    #                                                                     #
    #######################################################################

    imgs_clone = imgs.clone()
    lenx = imgs_clone.size(2)
    leny = imgs_clone.size(3)
    for i in range(imgs_clone.size(0)):
        for idx in range(0, lenx, tile_size):
            for idy in range(0, leny, tile_size):
                if np.random.random() < drop_rate:
                    for j in range(idx, idx + tile_size):
                        for k in range(idy, idy + tile_size):
                            imgs_clone[i, 0, j, k] = 0
    imgs_n = imgs_clone

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return imgs_n


# Task 1: evaluate learnt model on each denoising task
# => run entire program for each noise_type
# => Save a png for each encoded + decoded noise
# => Calculate accuracy of learnt model using the classifier but without finetuning (don't change learnt model)
# Task 2: what is impact of finetuning vs fixed feature representations and how does it change with dataset size?
# => run everything once without finetuning and once with finetuning
# => run with 3 different transfer dataset sizes: 5000, 2500, 1000
for finetune in [False, True]:
    # print('finetune:', finetune, '-->')
    for transfer_dataset_size in [5000, 2500, 1000]:
        # print('transfer_dataset_size:', transfer_dataset_size, '-->')
        # Choose transfer_dataset_size examples for transfer learning
        mask = np.random.randint(0, 60000, transfer_dataset_size)
        finetune_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              sampler=SubsetRandomSampler(np.where(mask)[0]),
                                              num_workers=2)

        for noise_type in noise_types:
            # print('run for noise type:', noise_type, '-->')
            encoder = Encoder()
            decoder = Decoder()
            if cuda_available:
                encoder = encoder.cuda()
                decoder = decoder.cuda()

            # Define Loss and Optimizer for DAE training
            parameters = list(encoder.parameters()) + list(decoder.parameters())
            loss_func = nn.MSELoss()
            optimizer = torch.optim.Adam(parameters, lr=learning_rate)

            # Get noise function to be applied to images
            if noise_type is 'gaussian_add':
                image_fn = noise_additive_gaussian
            elif noise_type is 'noise_salt_pepper':
                image_fn = noise_salt_pepper
            elif noise_type is 'noise_masking':
                image_fn = noise_masking
            else:
                # Default is no noise (standard AE)
                image_fn = lambda x: x

            # print('--------------------------------------------------------------')
            # print('---------------------- Training DAE --------------------------')
            # print('--------------------------------------------------------------')

            # Train the Autoencoder
            for epoch in range(num_epochs_autoencoder):
                losses = []
                start = time.time()
                for batch_index, (images, _) in enumerate(train_loader):
                    if cuda_available:
                        images = images.cuda()
                    images = Variable(images)
                    image_noised = image_fn(images)

                    # Training Step
                    optimizer.zero_grad()
                    output = encoder(image_noised)
                    output = decoder(output)
                    loss = loss_func(output, images)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.data[0])
                    #if batch_index % 50 == 0:
                    #    print('Epoch: {}, Iter: {:3d}, Loss: {:.4f}'.format(epoch, batch_index, loss.data[0]))

                end = time.time()
                # print('Epoch: {}, Average Loss: {:.4f}, Time: {:.4f}'.format(epoch, np.mean(losses), end - start))

            # Set encoder and decoder in evaluation mode to use running means and averages for Batchnorm
            encoder.eval()
            decoder.eval()

            if transfer_dataset_size == 5000 and not finetune and noise_type is not 'None':
                # save a plot only once for each noise type...
                # Get a batch of test images
                test_imgs, test_labels = next(iter(testloader))
                if cuda_available:
                    test_imgs, test_labels = test_imgs.cuda(), test_labels.cuda()
                test_imgs, test_labels = Variable(test_imgs), Variable(test_labels)
                test_imgs_noised = image_fn(test_imgs)

                output = encoder(test_imgs_noised)
                output = decoder(output)

                # Visualize in and output of the Autoencoder
                fig_out = plt.figure('out', figsize=(10, 10))
                fig_in = plt.figure('in', figsize=(10, 10))
                for ind, (img_out, img_in) in enumerate(zip(output, test_imgs_noised)):
                    if ind > 15:
                        break
                    plt.figure('out')
                    fig_out.add_subplot(4, 4, ind + 1)
                    plt.imshow(img_out.data.cpu().numpy().reshape(28, 28), cmap='gray')
                    plt.axis('off')
                    plt.figure('in')
                    fig_in.add_subplot(4, 4, ind + 1)
                    plt.imshow(img_in.data.cpu().numpy().reshape(28, 28), cmap='gray')
                    plt.axis('off')
                fig_in.savefig(noise_type + '-encoded.png')
                fig_out.savefig(noise_type + '-decoded.png')
                #plt.show()

            # print('--------------------------------------------------------------')
            # print('------------------- Transfer Learning ------------------------')
            # print('--------------------------------------------------------------')

            #######################################################################
            #                                                                #
            # Prepare everything for transfer learning:                           #
            #   - Build the classifier                                            #
            #   - Define the optimizer                                            #
            #   - Define the loss function                                        #
            # Note: The setup might be different for finetuning or fixed features #
            #       (see variable finetune!)                                      #
            #                                                                     #
            #######################################################################
            clf = Classifier()
            if cuda_available:
                clf = clf.cuda()
            if finetune:
                parameters = list(encoder.parameters()) + list(clf.parameters())
                encoder.train()
            else:
                parameters = clf.parameters()
            optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9)
            loss_func = nn.CrossEntropyLoss()

            #######################################################################
            #                         END OF YOUR CODE                            #
            #######################################################################

            # Train the Classifier
            best_test_accuracy = 0
            for epoch in range(num_epochs_classifier):
                losses = []
                start = time.time()
                for batch_index, (images, labels) in enumerate(finetune_loader):
                    if cuda_available:
                        images, labels = images.cuda(), labels.cuda()
                    images, labels = Variable(images), Variable(labels)

                    # Training Step
                    optimizer.zero_grad()
                    output = encoder(images)
                    output = clf(output)
                    loss = loss_func(output, labels)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.data[0])

                end = time.time()
                # print('Epoch: {}, Average Loss: {:.4f}, Time: {:.4f}'.format(epoch, np.mean(losses), end - start))

                #######################################################################
                #                                                                     #
                # Evaluate the classifier on the test set by computing the accuracy   #
                # of the classifier                                                   #
                #                                                                     #
                #######################################################################

                clf.eval()
                if finetune:
                    encoder.eval()
                batch_accuracies = []
                for batch_index, (images, labels) in enumerate(testloader):
                    if cuda_available:
                        images, labels = images.cuda(), labels.cuda()
                    images, labels = Variable(images), Variable(labels)

                    output = encoder(images)
                    prediction = clf(output)

                    prediction = torch.max(prediction, 1)[1]
                    correct = prediction.long().eq(labels).sum().data[0]

                    batch_size = labels.size(0)
                    batch_accuracies.append(correct * (100.0 / batch_size))

                accuracy = np.array(batch_accuracies).mean()

                if accuracy > best_test_accuracy:
                    best_test_accuracy = accuracy

                #######################################################################
                #                         END OF YOUR CODE                            #
                #######################################################################

                # print('Epoch: {}, Test Acc: {:.4f}'.format(epoch, accuracy))
                # print('--------------------------------------------------------------')
                clf.train()
                if finetune:
                    encoder.train()

            # print('run for noise type:', noise_type, '<-- [best test accuracy achieved:', best_test_accuracy, ']')
        # print('transfer_dataset_size:', transfer_dataset_size, '<--')
    # print('finetune:', finetune, '<--')
