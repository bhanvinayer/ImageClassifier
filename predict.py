import argparse
from torchvision import models
import torch
from torch import nn
import json
from collections import OrderedDict
import torch.nn.functional as F
from predictfunctions import loadmodel, process_image,  predict

def parse_predict_args():
    predict_parser = argparse.ArgumentParser()
    predict_parser.add_argument("image_path", type=str, help="path of image to be categorized")
    predict_parser.add_argument("checkpoint", type=str, help="checkpoint file path")
    predict_parser.add_argument("--top_k", type=int, default=3, help="no. of top K likely class")
    predict_parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="associating categories to names json file")
    predict_parser.add_argument("--gpu", action="store_true", help="if you want to use gpu for training the network (recommended)")

    return predict_parser.parse_args()

def print_top_k_results(class_names, probs, top_k):
    print("Top K classes:")
    for i in range(top_k):
        print(f"{class_names[i]}: Probability - {probs[i]:.4f}")

def main():
    args = parse_predict_args()
    gpu=args.gpu
    # Loading model
    model, epochs, class_to_idx = loadmodel(args.checkpoint)
    
    probability, classes = predict(args.image_path, model, args.top_k, gpu)

    # class names
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    # Mapping class indices to class names
    class_names = [cat_to_name[predict_class] for predict_class in classes]
    # Printing results
    print_top_k_results(class_names, probability, args.top_k)


if __name__ == "__main__":
    main()
