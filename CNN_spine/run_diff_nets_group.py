import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import copy
import argparse
import os
import torch
import torch.optim as optim
import logging
from torch.utils.tensorboard import SummaryWriter

# Polyaxon



class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def train_model(model, criterion, optimizer, scheduler, save_path, num_epochs,model_name,feature_extract, use_pretrained,dataset):


    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    writer = SummaryWriter(save_path)

    for epoch in range(num_epochs):
        # print('epoch:',epoch)
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, path in dataloaders[phase]:
                # print(inputs.shape)
                # print(labels.shape)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print(outputs.shape)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


                if epoch == 49:

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, os.path.join(save_path, "model" + str(epoch) + ".pt"))

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                writer.add_scalar('Loss/train', float(epoch_loss), epoch)
                writer.add_scalar('Accuracy/train', float(epoch_acc), epoch)
                # print(epoch_acc, 'accuracy')
                # print(epoch_loss,'loss')
            elif phase == 'val':
                writer.add_scalar('Loss/val', float(epoch_loss), epoch)
                writer.add_scalar('Accuracy/val', float(epoch_acc), epoch)
            # deep copy the model

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                }, os.path.join(save_path, "model_best_%s_fixed_%s_pretrained_%s_%s.pt" %(model_name,feature_extract, use_pretrained,dataset)))

    logger.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



def test_model(model, criterion, optimizer, scheduler, save_path, num_epochs):


    # best_model_wts = copy.deepcopy(model.state_dict())
    # best_acc = 0.0

    # writer = SummaryWriter(save_path)
    num_epochs = 1
    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.eval()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, path in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)



                # forward
                # track history if only in train

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)



                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                false_predictions = (preds == labels.data).cpu().detach().numpy()
                logger.info(path[1 - false_predictions]+'test model')
                # if epoch == 99:
                #
                #     torch.save({
                #         'epoch': epoch,
                #         'model_state_dict': model.state_dict(),
                #         'optimizer_state_dict': optimizer.state_dict(),
                #         'loss': loss,
                #     }, os.path.join(save_path, "model" + str(epoch) + ".pt"))

            # if phase == 'train':
            #     scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #     phase, epoch_loss, epoch_acc))
            #
            # if phase == 'train':
            #     writer.add_scalar('Loss/train', float(epoch_loss), epoch)
            #     writer.add_scalar('Accuracy/train', float(epoch_acc), epoch)
            # elif phase == 'val':
            #     writer.add_scalar('Loss/val', float(epoch_loss), epoch)
            #     writer.add_scalar('Accuracy/val', float(epoch_acc), epoch)
            # # deep copy the model

            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())

    # logger.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model

def initialize_model(model_name, num_classes, feature_extract, use_pretrained):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# plt.ioff()
# plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')

    parser.add_argument('--flag_Polyaxon', type= str, default="False",
                        help='TRUE - Polyaxon, False - CPU')

    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')

    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--dataset', type=str, default='data1', metavar='N',
                        help='cross validation dataset')

    parser.add_argument('--network', default='resnet', type=str, metavar='N',
                        help='resnet, alexnet, vgg, squeezenet, densenet, inception')

    parser.add_argument('--use_pretrained', default='True', type=str,
                        help='True - Imagenet initialized weights, False - train from scratch')

    parser.add_argument('--Feature_extractation', default='False',type=str,
                        help='True - fine tune last layer only, False - update weights')

    parser.add_argument('--info_experiment', type=str, default="False",
                        help='description of the experiment')

    args = parser.parse_args()

    Polyaxon_flag = args.flag_Polyaxon

    if Polyaxon_flag == "True":
        from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path
        data_paths = get_data_paths()
        outputs_path = get_outputs_path()
        data_dir = os.path.join(data_paths['data1'], "SpinousProcessData",'PolyU_dataset')
        dataset = args.dataset

        data_dir = os.path.join(data_dir, dataset)

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.FileHandler(os.path.join(outputs_path,'SpinousProcess.log')))

        # Polyaxon
        experiment = Experiment()

    else:
        data_dir = "/media/maryviktory/My Passport/spine navigation Polyu 2021/DATASET_polyu/PWH_sweeps/Subjects dataset/phantom_classes/"
        #'/media/maryviktory/My Passport/IPCAI 2020 TUM/DATA_toNas_for CNN_IPCAI/data_all(15patients train, 4 test)/'

        outputs_path = "/media/maryviktory/My Passport/spine navigation Polyu 2021/DATASET_polyu/PWH_sweeps/output/"
        #'/media/maryviktory/My Passport/IROS 2020 TUM/DATASETs/Dataset/DATA_toNas_for CNN_IPCAI/output/'
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.FileHandler(os.path.join(outputs_path, 'SpinousProcess.log')))



    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)


    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),

            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),

            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Use the image folder function to create datasets
    image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}

    # Make iterable objects with the dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    # need to preserve some information about our dataset,
    # specifically the size of the dataset and the names of the classes in our dataset.
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    num_classes = len(list(class_names))
    logger.info(str(class_names))

    #device - a CPU or GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Grab some of the training data to visualize
    inputs, classes, path = next(iter(dataloaders['train']))
    if Polyaxon_flag == "True":
        logger.info(path)
    else:
        print(path)

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
##########  Parse data
    model_name = args.network
    use_pretrained = args.use_pretrained
    if use_pretrained=="True":
        use_pretrained = True
    else:
        use_pretrained = False

    feature_extract = args.Feature_extractation
    if feature_extract=="True":
        feature_extract = True
    else:
        feature_extract = False

    dataset = args.dataset

    num_epochs = args.epochs


#######3 Chhose model

    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained)

    # print(model_ft)
    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    # print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                # print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)


    optimizer_ft = optim.Adam(model_ft.parameters(), lr=args.lr)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

    # Train and evaluate
    model_ft= train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, outputs_path, num_epochs, model_name,feature_extract, use_pretrained,dataset)


############### FINE TUNING #################

    # model_ft = models.densenet121(pretrained=True)
    # # print(model_ft)
    #
    #
    # #fc - fully connected layer, the last one
    # num_ftrs = model_ft.classifier.in_features
    # # Here the size of each output sample is set to 2.
    # # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    # # model_ft.fc = nn.Linear(num_ftrs, 2)
    # model_ft.classifier = nn.Linear(num_ftrs, 2)
    #
    #
    # for item in model_ft.parameters():
    #     item.requires_grad = False
    #     # logger.info(item + 'has been unfrozen.')
    #
    # model_ft = model_ft.to(device)
    #
    # criterion = nn.CrossEntropyLoss()
    #
    # # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.05, momentum=0.9)
    #
    # # Decay LR by a factor of 0.1 every 3 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)
    #
    # model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, outputs_path,
    #                        num_epochs=50)

    #
    # # visualize_model(model_ft)


############## FIXED FEATURE EXTRACTOR ###############
    # # Note that the parameters of imported models are set to requires_grad=True by default
    # model_conv = models.densenet121(pretrained=True)
    # for name, child in model_conv.named_children():
    #     if name in ['layer2','layer3', 'layer4','avgpool']:
    #         logger.info(name + 'has been unfrozen.')
    #         for param in child.parameters():
    #             param.requires_grad = True
    #     else:
    #         for param in child.parameters():
    #             param.requires_grad = False
    #
    # # Parameters of newly constructed modules have requires_grad=True by default
    # num_ftrs = model_conv.fc.in_features
    # model_conv.fc = nn.Linear(num_ftrs, 2)
    #
    # model_conv = model_conv.to(device)
    #
    # criterion = nn.CrossEntropyLoss()
    #
    # # Observe that only parameters of final layer are being optimized as
    # # opposed to before.
    # optimizer_conv = torch.optim.SGD(filter(lambda x: x.requires_grad, model_conv.parameters()), lr=0.0005, momentum=0.9)
    # # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    #
    # model_conv = train_model(model_conv, criterion, optimizer_conv,
    #                          exp_lr_scheduler,outputs_path, num_epochs=100)

