"""
    Run to train the model.
"""
import sys
import torch
from torch import nn
from torchsummary import summary

from model import OurAlexNet
from loader import MyDataset, readTrImages

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_summary():
    test = OurAlexNet()
    summary(test, input_size=(3, 224, 224))


def test_model(model, val_loader):
    print("    ")
    print("**************************STARTED VALIDATION CHECK**************************")
    total_correct = 0
    total_checked = 0
    for i, data in enumerate(val_loader):
        # get the inputs; data is a list of [batch, labels]
        batch, labels = data
        batch = batch.float()
        labels = labels.float()
        outputs = model(batch)
        num_correct = torch.sum(torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1)).item()
        total_correct += num_correct 
        total_checked += len(outputs)
    percentage_corr = total_correct/total_checked
    print("PERCENTAGE CORRECT IS " + str(percentage_corr) + "%")
    print("**************************ENDED VALIDATION CHECK**************************")
    return percentage_corr


def train(train_loader, val_loader, num_epochs, loadModel=False, loadFN=None):
    model = OurAlexNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-2,momentum=0.9,weight_decay=0.0005)
    best_loss = 100000000
    best_val_loss = 0
    best_model = OurAlexNet()
    start_epoch = 0

    if loadModel:
        checkpoint = torch.load(loadFN)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        best_val_loss = checkpoint['vloss']
        print("Loaded Model")

    num_iters = len(train_loader)

    for epoch in range(start_epoch, num_epochs):  # loop over the dataset multiple times  
        print("Training epoch: " + str(epoch) + " of " + str(num_epochs))
        running_loss = 0.0
        
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [batch, labels]
            print("Started Iteration: " + str(i) + " of " + str(num_iters))
            batch, labels = data
            batch = batch.float()
            labels = labels.float()
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            running_loss += loss       
            print("Backward done on this iteration")
            optimizer.step()
            print("Model finished running iteration", i, "\n")
            print("Running Loss:", running_loss)

        # Validation Loss
        val_loss = test_model(model, val_loader)

        if running_loss < best_loss or val_loss > best_val_loss:
            best_loss = running_loss
            best_model = model
            best_val_loss = val_loss
            model_fn = "./models/model_epoch" + str(epoch) + "_loss" + str(best_loss) + "_vloss" + str(val_loss) + ".pt"
            torch.save({
            'epoch': (epoch+1),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss,
            'vloss': val_loss
            }, model_fn)

        print("Finished Running Epoch ", epoch, " loss: ", loss, "\n")

    print('Finished Training')
    return best_model

if __name__ == '__main__':
    train_dataset, train_loader, test_dataset, val_loader = readTrImages(8, 0.7, dim=224)
    if len(sys.argv) > 1:
        model = train(train_loader, val_loader, 100, loadModel=True, loadFN=sys.argv[1])
    else:
        model = train(train_loader, val_loader, 100, loadModel=False, loadFN=None)
