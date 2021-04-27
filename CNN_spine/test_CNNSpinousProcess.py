from rospine_utils import *
from torchvision import transforms
import torch
import os
from PIL import Image

# test_dir = "/media/maryviktory/My Passport/IPCAI 2020 TUM/CNN/data_all(15patients train, 4 test)/test"
test_dir = "/media/maryviktory/My Passport/IROS 2020 TUM/DATASETs/Dataset/PWH_sweeps/Subjects dataset/sweep3005/Classes"
model_path = "/media/maryviktory/My Passport/IROS 2020 TUM/DATASETs/Dataset/PWH_sweeps/model_best_resnet_fixed_False_pretrained_True_PolyU_dataset.pt"

def list2batch(tensor_list, label_list):

    if label_list is not None and len(label_list) == 0:
        label_list = None
    if label_list is not None:
        assert len(tensor_list) == len(label_list), "Number of labels files do not match number of images files"

    tensor_batch = torch.FloatTensor()
    torch.cat(tensor_list, out=tensor_batch)

    if label_list is None:
        return tensor_batch, None

    if type(label_list[0]) is torch.Tensor:
        label_batch = torch.FloatTensor()
        torch.cat(label_list, out=label_batch)

    else:
        label_batch = torch.FloatTensor(label_list)

    return tensor_batch, label_batch


def main():
    model = ModelLoader_types(model_path, model_type = "classification")
    model.eval()
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
            image_batch, _ = list2batch([tensor_image], None)

            out = model.run_inference(image_batch)
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


    print('correct',n_correct, n_correct/n_total)
    print("true positive", n_tp,n_tp/n_total)
    print("false positive", n_fp, n_fp / n_total)
    print("false negative", n_fn, n_fn / n_total)
    print("true negative", n_tn, n_tn / n_total)
    print('total',n_total)




if __name__ == '__main__':
    main()
