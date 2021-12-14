"""
    Run this file with a required argument - path to model, and this will output a submission file.
"""
import os
import sys
import torch
from torchvision import transforms
from PIL import Image
from torch import nn
# from model import OurAlexNet
from model import OurAlexNet, faster_rcnn, BB_model
from loader import readTrImages

def runModel(model, img_fn, dim):
    image = Image.open(img_fn)
    image = image.resize((dim, dim))
    convert_tensor = transforms.ToTensor()
    x = convert_tensor(image)
    x = torch.stack([x])
    x = x.float()
    model.eval()
    outputs = model(x)[0]
    # print(outputs)
    # assert False
    y = torch.argmax(outputs, dim=1).item()
    print(y)
    return y

def loadModel(model_fn = None):
    # loads the model
    model = BB_model()
    if model_fn == None:
        return model
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001, betas=(0.9, 0.999))
    checkpoint = torch.load(model_fn)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['loss']
    print("Loaded Model")
    return model

def writeSubmission(model_fn = None, dim=224):
    # The code for the submission after running the model.
    output = open('Team19.txt', 'w')
    output.write('guid/image,label\n')
    count = 0

    model = loadModel(model_fn)

    # loader, _ = readTrImages(1, 1)

    labels = []

    # for i, data in enumerate(loader):
    #     batch, labels = data
    #     batch = batch.float()
    #     with torch.no_grad():
    #         outputs = model(batch)[0]
    #     label = torch.argmax(outputs, dim=1).item()
    #     labels.append(label)


    count = 0
    folder_count = 1
    for folder in os.listdir('./deploy/test/'):
        print(folder)
        print(folder_count)
        for file in os.listdir('./deploy/test/' + folder):
            if file.endswith("_image.jpg"):
                tmp_fn = './deploy/test/' + folder + "/" + file
                out_label = runModel(model, tmp_fn, dim)
                # out_label = labels[count]
                output.write(folder + "/" + file[:-10] + "," + str(out_label) + "\n")
                count += 1
        # assert False
        # if folder_count == 2:
        #     assert False
        folder_count+=1

    print(count)
    output.close()

if __name__ == '__main__':
    # model_fn = sys.argv[1]
    # writeSubmission(model_fn)
    writeSubmission()
