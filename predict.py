import os
import json

import cv2
import numpy as np

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from vit_model import vit_base_patch16_224_in21k as create_model



def main():
    class_indict = {
                "0": 'blackcherry',
                "1": 'butternut',
                # "1": 'chestnut',
                # "2": 'redoak',
                # "4": "redpine",
                "2": 'walnut',
                "3": "wahitoak",
                # "6": "whitepine",
    }
    label_list=[]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

     # read class_indict
    # json_path = './class_indices.json'
    # assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    # with open(json_path, "r") as f:
    #     class_indict = json.load(f)

    # create model
    model = create_model(num_classes=4, has_logits=False).to(device)
    # load model weights
    model_weight_path = "F:/course/tree_identification/training_weights/vit_weights_20220316/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # load image
    predict_dir = r"F:/course/tree_identification/dataset/1116test_for0316"
    assert os.path.exists(predict_dir), "file: '{}' dose not exist.".format(predict_dir)
    test = os.listdir(predict_dir)
    for file in os.listdir(predict_dir):
        filepath=os.path.join(predict_dir,file)
        image = Image.open(filepath)
        # plt.imshow(img)
        image = data_transform(image)
        # expand batch dimension
        image = torch.unsqueeze(image, dim=0)

        with torch.no_grad():
        # predict class
            output = torch.squeeze(model(image.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            label_list.append(class_indict[str(predict_cla)])
        
     
    # print(class_indict[str(predict_cla)])
    print(label_list)




if __name__ == '__main__':
    main()
