### FIND the probabilities of Classes, "Gap", "NonGap"
### p_c1 - non Gap
### p_c2 - Gap

import rospine_utils as utils
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas
import os
from torch import nn

patients = []


def processing_thread(model, input_list, label_list):
    inputs, labels = utils.list2batch(input_list, label_list)
    output = model.run_inference(inputs.to("cpu"))

    # with torch.no_grad():
    #     output = model.forward(inputs)

    prob = torch.sigmoid(output)

    c1 = prob[0, 0]
    c2 = prob[0, 1]

    c1_array = np.squeeze(c1.to("cpu").numpy())
    c2_array = np.squeeze(c2.to("cpu").numpy())

    labels = float(labels.to("cpu").item())

    return c1_array.tolist(), c2_array.tolist(), labels

def open_file(name_file):
    train_file = open(name_file, "r")
    lines_train = train_file.read().splitlines()
    print('patient names:',lines_train)
    return lines_train



##############___Load model ##############

model_path = "/media/maryviktory/My Passport/IPCAI 2020 TUM/CNN data_scripts/models/facet_model_18_finetune.pt"

# model = models.resnet18(pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 2)
#
# checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
# print(checkpoint)
# model.load_state_dict(checkpoint['model_state_dict'])


model = utils.ModelLoader(model_path)
model.to_device("cpu")
transformation = transforms.Compose([transforms.Resize(256),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])
model.eval()

###########____Load_Patients ######


# patients = open_file('/home/maria/IFL_project/forceultrasuondspine/Spinous_counting/dataset_to_plot.txt')
path = "/media/maryviktory/My Passport/IPCAI 2020 TUM/Facet_Db/test/"

for patient in os.listdir(os.path.join(path,"images")):
    patients.append(patient)




for i in range(len(patients)):

    patient_name = patients[i]
    loader_type = "offline"

    buffer_len = 1
    data_path = path
    sweep_folder = os.path.join(path,'images',patient_name)
    print(sweep_folder)
    sweep_label_folder = os.path.join(path,'labels',patient_name)
    sort_type = 0
    bias = 0
    save_folder = os.path.join(path,'csv')
    pd_frame = pandas.DataFrame(columns=['Facet Probability', 'Label'])

    loader = utils.OfflineLoader(sweep_path=sweep_folder,
                                sweep_label_path=sweep_label_folder,
                                sort_type=sort_type,
                                bias=bias)


    # Initializing processing thread
    tensor_list, label_list = [], []
    p_c1, p_c2 = [], []
    plt.ion()


    while True:

        current_image, current_label = loader.get_next_image()
        if current_image is None:
            break
        tensor_list.append(transformation(current_image).unsqueeze_(0))
        label_list.append(current_label)

        if len(tensor_list) >= buffer_len:
            c1, c2, labels_list = processing_thread(model, tensor_list, label_list)
            p_c1.append(c1)
            p_c2.append(c2)


            pd_frame = pd_frame.append({'Facet Probability': c1, 'Label': labels_list},
                                       ignore_index=True)

            pd_frame.to_csv(os.path.join(data_path, patient_name+"_Facet_Probability_label.csv"))


            # print(p_c1)
            # print(label_list)
            tensor_list, label_list = [], []

    print(patient_name,'  ready')