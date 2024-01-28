import torch
from torchvision import models, transforms, datasets
from torch import nn, optim
from collections import OrderedDict

def load_data(data_dir,gpu):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    if gpu==True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])



    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, 
                                              batch_size=64, 
                                              shuffle=True, 
                                              pin_memory = True  if str(device) == "cuda" else 0)

    validloader = torch.utils.data.DataLoader(validation_data,
                                              batch_size =32,
                                              shuffle = True,
                                              pin_memory = True  if str(device) == "cuda" else 0)

    testloader = torch.utils.data.DataLoader(test_data,
                                             batch_size = 20,
                                             shuffle = True,
                                             pin_memory = True  if str(device) == "cuda" else 0)
    print("Data loaded successfully")
    return trainloader, validloader, testloader, train_data
# TODO: Build and train your network
# Pre-Trained architecture usage
def pretrained_model(model_name='resnet18'):
    if model_name.lower() == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_name.lower() == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model_name.lower() == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model

def train_model(model, trainloader, validloader, learning_rate, epochs, gpu, hiddenunits):
    for param in model.parameters():
        param.requires_grad = False
    
    # Defining the classifier with layers  
    if isinstance(model, models.resnet.ResNet):
        classifier = nn.Sequential(OrderedDict([
            ('inputs', nn.Linear(512, hiddenunits)),
            ('relu1', nn.ReLU()),
            ('hidden_layer1', nn.Linear(hiddenunits, 128)),
            ('relu2', nn.ReLU()),
            ('hidden_layer2', nn.Linear(128, 120)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(), lr = 1e-3)
    else:
        classifier = nn.Sequential(OrderedDict([
            ('inputs', nn.Linear(model.classifier[0].in_features, hiddenunits)),
            ('relu1', nn.ReLU()),
            ('hidden_layer1', nn.Linear(hiddenunits, 128)),
            ('relu2', nn.ReLU()),
            ('hidden_layer2', nn.Linear(128, 120)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr = 1e-3)
        
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # defining criterion 
    criterion = nn.NLLLoss()
    
    # defining number of epochs
    num_epochs = epochs

    for epoch in range(num_epochs):
        # defining training and validation loss
        running_loss = 0.0
        correct_train_predictions = 0
        total_train_samples = 0

        valid_running_loss = 0.0
        correct_valid_predictions = 0
        total_valid_samples = 0

        # setting model to training mode
        model.train()

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            labels = labels.to(device)
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # Calculatation of training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train_samples += labels.size(0)
            correct_train_predictions += (predicted == labels).sum().item()

        # setting model to evaluation mode
        model.eval()

        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)

                valid_running_loss += loss.item() * inputs.size(0)

                # Calculatation of validation accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_valid_samples += labels.size(0)
                correct_valid_predictions += (predicted == labels).sum().item()

        # calculatation of average loss and accuracy
        train_loss = running_loss / len(trainloader.dataset)
        valid_loss = valid_running_loss / len(validloader.dataset)

        train_accuracy = correct_train_predictions / total_train_samples
        valid_accuracy = correct_valid_predictions / total_valid_samples

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Training Loss: {train_loss:.3f} | Training Accuracy: {train_accuracy:.3f} |")
        print(f"Validation Loss: {valid_loss:.3f} | Validation Accuracy: {valid_accuracy:.3f}")
    
    print("model trained successfully")
    return model, optimizer

def save_checkpoint(model, optimizer, epochs, checkpoint_path, arch, hiddenunits):
    # TODO: Save the checkpoint 
    #Defining a dictionary for checkpoint to save the network
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs,
        'arch': arch,
        'hiddenunits':hiddenunits
    }

    # Saving the checkpoint to the file
    torch.save(checkpoint, checkpoint_path)

    print(f"Checkpoint saved to", checkpoint_path)