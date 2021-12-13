"""
    Dataloaders for training and testing.
    Import the function readTrImages and MyDataset object for creating the training and validation dataloaders.
    Important!! - Make sure the deploy folder is in the same directory as this loader file. 
    TODO: Make testing code.       
"""
import sys
import os
import struct
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import random
from numpy.lib.shape_base import split
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
from matplotlib.pyplot import imshow
from matplotlib import cm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

classes = (
    'Unknown', 'Compacts', 'Sedans', 'SUVs', 'Coupes',
    'Muscle', 'SportsClassics', 'Sports', 'Super', 'Motorcycles',
    'OffRoad', 'Industrial', 'Utility', 'Vans', 'Cycles',
    'Boats', 'Helicopters', 'Planes', 'Service', 'Emergency',
    'Military', 'Commercial', 'Trains'
)
classes_to_labels = {
    'Unknown':0, 'Compacts':1, 'Sedans':1, 'SUVs':1, 'Coupes':1,
    'Muscle':1, 'SportsClassics':1, 'Sports':1, 'Super':1, 'Motorcycles':2,
    'OffRoad':2, 'Industrial':2, 'Utility':2, 'Vans':2, 'Cycles':2,
    'Boats':0, 'Helicopters':0, 'Planes':0, 'Service':0, 'Emergency':0,
    'Military':0, 'Commercial':0, 'Trains':0
}

class MyDataset(Dataset):
    # An object for representing the Dataset for Pytorch.
    def __init__(self, image_fns, labels, dim):
        super().__init__()
        self.image_fns = image_fns
        self.image_orig_shape = None
        self.labels = labels
        if not dim:
            self.dim = False
        else:
            self.dim = (dim, dim)

    def __len__(self):
        return max(len(self.image_fns), len(self.labels))

    def getImage(self, index):
        image = Image.open(self.image_fns[index])
        self.image_orig_shape = image.size
        if self.dim:
            image = image.resize(self.dim)
        convert_tensor = transforms.ToTensor()
        image = torch.tensor(convert_tensor(image), requires_grad=True)
        return image
    
    def getMask(self, index):
        cols,rows = self.image_orig_shape
        bb = self.labels[index][0][1]
        label = self.labels[index][0][0][0]
        label = classes_to_labels[label]
        Y = np.zeros((rows, cols))
        bb = np.array(bb)
        bb = bb.astype(np.int)
        Y[bb[0]:bb[1], bb[1]:bb[2]] = 1.
        bb = [bb[2], bb[0], bb[3], bb[1]]
        mask = Image.fromarray(np.uint8(cm.gist_earth(Y)*255))
        if self.dim:
            mask = mask.resize(self.dim)
        convert_tensor = transforms.ToTensor()
        mask = torch.tensor(convert_tensor(mask), requires_grad=True)
        label = torch.tensor(label, dtype=torch.long)
        return label, mask, bb

    def __getitem__(self, index):
        image = self.getImage(index)
        label, mask, bb = self.getMask(index)
        target = {}
        target["boxes"] = bb
        target["labels"] = label
        target["masks"] = mask
        return image, target

def writeFirstSubmission():
    # The code for the first submission.
    output = open('Team19.txt', 'w')
    output.write('guid/image,label\n')
    count = 0

    for folder in os.listdir('./deploy/test/'):
        for file in os.listdir('./deploy/test/' + folder):
            if file.endswith("_image.jpg"):
                output.write(folder + "/" + file[:-10] + ",1\n")
                count += 1

    print(count)
    output.close()
    
def readTrLabels(train=False):
    # Reading the training labels from the generated trainval_labels.csv file, which is generated by extract_info.py from Canvas.
    file_to_read = "deploy/trainval/trainval_labels.csv"
    label_dict = {}
    file_data = open(file_to_read)
    count = 0
    for row in file_data:
        if count != 0:
            split_lst = row.split(',')
            label_dict[split_lst[0]] = int(split_lst[1][0])
        count += 1
    return label_dict

def rot(n):
    n = np.asarray(n).flatten()
    assert(n.size == 3)

    theta = np.linalg.norm(n)
    if theta:
        n /= theta
        K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

        return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    else:
        return np.identity(3)

def get_bbox(p0, p1):
    """
    Input:
    *   p0, p1
        (3)
        Corners of a bounding box represented in the body frame.

    Output:
    *   v
        (3, 8)
        Vertices of the bounding box represented in the body frame.
    *   e
        (2, 14)
        Edges of the bounding box. The first 2 edges indicate the `front` side
        of the box.
    """
    v = np.array([
        [p0[0], p0[0], p0[0], p0[0], p1[0], p1[0], p1[0], p1[0]],
        [p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
        [p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]
    ])
    e = np.array([
        [2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],
        [7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]
    ], dtype=np.uint8)

    return v, e

def readBbox(snapshot):
    xyz = np.fromfile(snapshot.replace('_image.jpg', '_cloud.bin'), dtype=np.float32)
    xyz = xyz.reshape([3, -1])

    proj = np.fromfile(snapshot.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
    proj.resize([3, 4])

    try:
        bbox = np.fromfile(snapshot.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
    except FileNotFoundError:
        print('[*] bbox not found.')
        bbox = np.array([], dtype=np.float32)
    
    bbox = bbox.reshape([-1, 11])

    uv = proj @ np.vstack([xyz, np.ones_like(xyz[0, :])])
    uv = uv / uv[2, :]

    bbox_list = []
    
    for k, b in enumerate(bbox):
        R = rot(b[0:3])
        t = b[3:6]

        sz = b[6:9]
        vert_3D, edges = get_bbox(-sz / 2, sz / 2)
        vert_3D = R @ vert_3D + t[:, np.newaxis]

        vert_2D = proj @ np.vstack([vert_3D, np.ones(vert_3D.shape[1])])
        vert_2D = vert_2D / vert_2D[2, :]
        min_x, min_y = 10000, 10000
        max_x, max_y = -1000, -1000

        for e in edges.T:
            #print(vert_2D[0, e], vert_2D[1, e]) # 2D bbox
            x1 = vert_2D[0, e][0]
            x2 = vert_2D[0, e][1]
            y1 = vert_2D[1, e][0]
            y2 = vert_2D[1, e][1]

            if y1 < min_y and y1 > 0:
                min_y = int(y1)
            if y1 > max_y:
                max_y = int(y1)
            if y2 < min_y and y2 > 0:
                min_y = int(y2)
            if y2 > max_y:
                max_y = int(y2)

            if x1 < min_x and x1 > 0:
                min_x = int(x1)
            if x1 > max_x:
                max_x = int(x1)
            if x2 < min_x:
                min_x = int(x2)
            if x2 > max_x:
                max_x = int(x2)
                       
            #print(vert_3D[0, e], vert_3D[1, e], vert_3D[2, e]) # 3d bbox

        c = classes[int(b[9])]
        bbox_list.append([[c], [min_y, max_y, min_x, max_x]])
    
    return bbox_list

def createDataset(images, labels, batch_size, dim):
    # Making dataset and dataloaders for PyTorch from a set of image filenames and labels.
    dataset = MyDataset(images ,labels, dim)
    loader = DataLoader(dataset,batch_size=batch_size, shuffle=False)
    return dataset, loader

def readTrImages(batch_size, split_ratio, dim=256, shuffle=False, train=False):
    # Reading and splitting the images from trainval and returning the dataset objects and the dataloaders for training.
    #labels_dict = readTrLabels()
    images = []
    labels = []

    for folder in os.listdir('./deploy/trainval/'):
        if folder.endswith(".csv"):
            continue
        for file in os.listdir('./deploy/trainval/' + folder):
            if file.endswith("_image.jpg"):
                image_id = folder + "/" + file[:-10]
                #tmp_label = labels_dict[image_id]
                tmp_fn = './deploy/trainval/' + folder + "/" + file
                images.append(tmp_fn)
                #labels.append(tmp_label)
                bbox_list = readBbox(tmp_fn)
                labels.append(bbox_list)
                                             

    print(str(len(images)) + " images read.")
    print(str(len(labels)) + " labels read.")
    idx_shuffled = list(range(0, len(images)))

    if shuffle:
        random.shuffle(idx_shuffled)

    split_idx = int(split_ratio * len(images))
    train_images = [images[i] for i in idx_shuffled[:split_idx]]
    train_labels = [labels[i] for i in idx_shuffled[:split_idx]]
    val_images = [images[i] for i in idx_shuffled[split_idx:]]
    val_labels = [labels[i] for i in idx_shuffled[split_idx:]]

    train_dataset, train_loader = createDataset(train_images, train_labels, batch_size, dim)

    val_dataset, val_loader = createDataset(val_images, val_labels, batch_size, dim)
    print("Dataloaders created, train has " + str(len(train_dataset)) + " samples and val has " + str(len(val_dataset)) + " samples.")
    return train_loader, val_loader

# To test
if __name__ == '__main__':
    train_loader, val_loader = readTrImages(8, 0.7, dim=False)

