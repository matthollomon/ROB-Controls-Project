"""
    Run to train the model.
"""
import sys
import torch
from torch import nn
from torchsummary import summary
import torchvision
import torch.nn.functional as F

from model import OurAlexNet, faster_rcnn, BB_model
from loader import MyDataset, readTrImages

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_summary():
    test = OurAlexNet()
    summary(test, input_size=(3, 224, 224))


def test_model(model, val_loader):
    model.eval()
    print("    ")
    print("**************************STARTED VALIDATION CHECK**************************")
    total_correct = 0
    total_checked = 0
    for i, data in enumerate(val_loader):
        # get the inputs; data is a list of [batch, labels]
        batch, labels = data
        batch = batch.float()
        with torch.no_grad():
            outputs = model(batch)
        # print(outputs)
        # print(labels['labels'])
        num_correct = torch.sum(torch.argmax(outputs[0], dim=1) == labels['labels']).item()
        total_correct += num_correct 
        total_checked += len(outputs[0])
    percentage_corr = total_correct/total_checked
    print("PERCENTAGE CORRECT IS " + str(percentage_corr) + "%")
    print("**************************ENDED VALIDATION CHECK**************************")
    return percentage_corr


def train(train_loader, val_loader, num_epochs, loadModel=False, loadFN=None):
    model = BB_model()
    criterion = nn.CrossEntropyLoss()
    lr = 1e-3
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    best_loss = 100000000
    best_val_acc = 0
    best_model = BB_model()
    start_epoch = 0
    C = 1000

    if loadModel:
        checkpoint = torch.load(loadFN)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        best_val_acc = checkpoint['vacc']
        print("Loaded Model")

    num_iters = len(train_loader)
    # val_acc = test_model(model, val_loader)

    for epoch in range(start_epoch, num_epochs):  # loop over the dataset multiple times  
        print("Training epoch: " + str(epoch) + " of " + str(num_epochs))
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader):

            # get the inputs; data is a list of [batch, labels]
            print("Started Iteration: " + str(i) + " of " + str(num_iters))
            batch, targets = data
            y_class = targets["labels"]
            y_bb = targets["boxes"]
            out_class, out_bb = model(batch)
            loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
            loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
            loss_bb = loss_bb.sum()
            loss = loss_class + loss_bb/C
            #loss = criterion(outputs, targets)
            loss.backward()
            running_loss += loss       
            print("Backward done on this iteration, learning rate - ", lr)
            optimizer.step()
            print("Model finished running iteration", i, "\n")
            print("Loss:", loss)
            print("Running Loss:", running_loss)
            


        # Validation Loss
        val_acc = test_model(model, val_loader)

        if running_loss < best_loss or val_acc > best_val_acc:
            best_loss = running_loss
            best_model = model
            best_val_acc = val_acc
        model_fn = "./models/model_epoch_new_model_" + str(epoch) + "_loss" + ".pt"
        torch.save({
        'epoch': (epoch+1),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss,
        'vacc': val_acc
        }, model_fn)

        print("Finished Running Epoch ", epoch, " loss: ", loss, "\n")

    print('Finished Training')
    return best_model

if __name__ == '__main__':
    train_loader, val_loader = readTrImages(16, 0.7, dim=200)
    if len(sys.argv) > 1:
        model = train(train_loader, val_loader, 100, loadModel=True, loadFN=sys.argv[1])
    else:
        model = train(train_loader, val_loader, 100, loadModel=False, loadFN=None)
