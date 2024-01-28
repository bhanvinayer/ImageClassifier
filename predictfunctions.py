import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image
from collections import OrderedDict
import torch.nn.functional as F

def loadmodel(filepath):
    checkpoint = torch.load(filepath)
    
    arch = checkpoint.get('arch', 'resnet18')
    hiddenunits = checkpoint.get('hiddenunits')
    
    if arch.lower() == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch.lower() == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif arch.lower() == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {arch}")    
        
    for param in model.parameters():
        param.requires_grad = False
    
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
           
    model.load_state_dict(checkpoint['model_state_dict'])
    epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer_state_dict']
    class_to_idx = checkpoint['class_to_idx']
    model.class_to_idx = class_to_idx
    print("model loaded successfully")
    
    return model, epochs, optimizer

def process_image(image):
   
    resize_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Opening the image using PIL
    image = Image.open(image)
    
    # Applying the transformations
    transformed_image = resize_image(image)
    
    return transformed_image

def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    
    # Loading and preprocessing the input image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    image = Image.open(image_path)
    input_image = transform(image).unsqueeze(0)
    
    input_image = input_image.to(device)
    # Making predictions
    with torch.no_grad():
        output = model(input_image)
        
    output = output.cpu()
    # Converting output to probabilities and class indices
    probabilities, indices = torch.topk(F.softmax(output, dim=1), topk)
    
    # Converting class indices to class labels
    class_to_idx_inv = {v: k for k, v in model.class_to_idx.items()}
    classes = [class_to_idx_inv[idx.item()] for idx in indices[0]]
    
    return probabilities.numpy()[0], classes
    
