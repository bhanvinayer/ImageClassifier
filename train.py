import argparse
from torchvision import datasets, transforms, models
import torch
from torch import nn, optim
from collections import OrderedDict
from trainfunctions import pretrained_model,load_data, train_model, save_checkpoint


def parse_train_args():
    train_parser=argparse.ArgumentParser()
    #epochs, datafile, learning rate, arch, gpu if available
    train_parser.add_argument("data_dir", type=str, help="give path for data file- flowers in this case")
    train_parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pth", help="File to save the model for making checkpoint and loading later")
    train_parser.add_argument("--arch", type=str, default="resnet18", choices=['vgg16', 'resnet18', 'alexnet'], help="Model architecture choices: 'vgg16', 'resnet18', or 'alexnet'")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for training th network")
    train_parser.add_argument("--epochs", type=int, default=6, help="no. of epochs for training the model")
    train_parser.add_argument("--hiddenunits", type=int, default=256, help="no.of hidden units for input size and layers of the network")
    train_parser.add_argument("--gpu", action="store_true", help="if you want to use gpu for training the network (recommended)")
    
    return train_parser.parse_args()

def main():
    args = parse_train_args()
    gpu=args.gpu
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    trainloader, validloader, testloader, train_data = load_data(args.data_dir,args.gpu)
    model = pretrained_model(args.arch)
    #optimizer=optim.Adam(model.fc.parameters(), lr=args.lr)
    model, optimizer = train_model(model, trainloader, validloader, args.lr, args.epochs, args.gpu, args.hiddenunits)
    model.class_to_idx = train_data.class_to_idx
    save_checkpoint(model, optimizer, args.epochs, args.checkpoint_path, args.arch, args.hiddenunits)

if __name__ == "__main__":
    main()