from rospine_utils import *
from torchvision import transforms, models
import torch
import os
import logging
from torch import nn

test_dir = "/media/maryviktory/My Passport/IPCAI 2020 TUM/DATA_toNas_for CNN_IPCAI/data_all(15patients train, 4 test)/test"
path = "/media/maryviktory/My Passport/IPCAI 2020 TUM/CNN data_scripts/models/densenet_vgg/"
log_path = "/media/maryviktory/My Passport/IPCAI 2020 TUM/CNN data_scripts/models/densenet_vgg/"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler(os.path.join(log_path, 'Probabilities.log'),mode= 'w'))

num_classes = 2
mode = "resnet"

path = os.path.join(path,mode)
use_pretrained = True
feature_extract = False

# model_ft = models.densenet121(pretrained=use_pretrained)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
def main():

    model_ft = None

    if mode == "resnet":
        model_ft = models.resnet18(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)


    elif mode == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224


    elif mode == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224


    print("model loaded %s"%(mode))

    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH, map_location=device))


    # checkpoint = ModelLoader(model_path)
    # model_ft.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model_ft.load_state_dict(checkpoint['model_state_dict'])

    # model_ft = ModelLoader(mode, use_pretrained,feature_extract, num_classes = 2, ckpt_path=model_path)

    model_ft.eval()
    transformation = transforms.Compose([transforms.Resize(256),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])

    gap_list = os.listdir(os.path.join(test_dir, "Gap"))
    non_gap_list = os.listdir(os.path.join(test_dir, "NonGap"))

    gap_list = [os.path.join(test_dir, "Gap", item) for item in gap_list]
    non_gap_list = [os.path.join(test_dir, "NonGap", item) for item in non_gap_list]

    n_correct = 0
    n_total = 0
    n_tp = 0
    n_fn = 0
    n_fp = 0
    n_tn = 0



    for i, test_db in enumerate([gap_list, non_gap_list]):
        for data in test_db:
            input_data = Image.open(data).convert(mode="RGB")
            if input_data is None:
                return input_data
            tensor_image = transformation(input_data).unsqueeze_(0)
            image_batch, _ = utils.list2batch([tensor_image], None)

            # out = model_ft.run_inference(image_batch)
            with torch.no_grad():
                out = model_ft.forward(image_batch)

            prob = torch.sigmoid(out).numpy()
            # print('prob',prob)
            # print('i',i)

            prob_vertebra = prob[0, 1]
            prob_gap = prob[0, 0]
            # print(prob_gap)
            # print(prob)

            # print('prob vert',prob_vertebra)
            if round(prob_vertebra) == i:
                n_correct += 1


            ###truepositive
            if round(prob_vertebra)== 1 and i == 1:
                n_tp +=1

            ###truenegative
            if round(prob_vertebra)== 0 and i == 0:
                n_tn +=1

            # false negative
            if round(prob_vertebra)==0 and i == 1:
                n_fn +=1

            ## false positive
            if round(prob_vertebra) == 1 and i == 0:
                n_fp += 1

            n_total += 1

    logger.info("_____________________________________")
    logger.info('correct {},{}'.format(n_correct, n_correct / n_total))
    logger.info( "true positive {},{}".format(n_tp, n_tp / n_total))
    logger.info("false positive {},{}".format(n_fp, n_fp / n_total))
    logger.info("false negative {},{}".format(n_fn, n_fn / n_total))
    logger.info("true negative {},{}".format(n_tn, n_tn / n_total))
    logger.info('total {}'.format(n_total))
    logger.info("_____________________________________")

    print('correct ',n_correct, n_correct/n_total)
    print("true positive", n_tp,n_tp/n_total)
    print("false positive", n_fp, n_fp / n_total)
    print("false negative", n_fn, n_fn / n_total)
    print("true negative", n_tn, n_tn / n_total)
    print('total',n_total)


for folder in os.listdir(path):
    print('folders',folder)
    logger.info('folder {}'.format(folder))

    if folder == "fine-tune":
        use_pretrained = True
        feature_extract = False

    elif folder == "last_layer":
        use_pretrained = True
        feature_extract = True

    elif folder == "scratch":
        use_pretrained = False
        feature_extract = False

    for model in os.listdir(os.path.join(path,folder)):
        print('models',model)
        logger.info('models:{:}'.format(model))
        model_path = os.path.join(path,folder,model)
        print('path for model',model_path)
        logger.info('models:{:}'.format(model_path))
        main()
        # print('next')

# if __name__ == '__main__':
#     main()
