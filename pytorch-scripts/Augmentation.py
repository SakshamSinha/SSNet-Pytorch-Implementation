''' Module for data augmentation. Two strategies have been demonstrated below. 
You can check for more strategies at 
http://pytorch.org/docs/master/torchvision/transforms.html '''

from torchvision import transforms
import torch


class Augmentation:   
    def __init__(self,strategy):
        print ("Data Augmentation Initialized with strategy %s"%(strategy));
        self.strategy = strategy;
        
        
    def applyTransforms(self):
        if self.strategy == "H_FLIP": # horizontal flip with a probability of 0.5
            data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        elif self.strategy == "H_FLIP_ROTATE": # horizontal flip with a probability of 0.5
            data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(15,15,15),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }


        elif self.strategy == "SCALE_H_FLIP": # resize to 224*224 and then do a random horizontal flip.
            data_transforms = {
            'train': transforms.Compose([
                transforms.Scale([224,224]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Scale([224,224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        elif self.strategy == "SCALE_H_FLIP_CROP": # resize to 224*224 and then do a random horizontal flip.
            
            normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            toTensor = transforms.ToTensor()
            data_transforms = {
                'train': transforms.Compose([
                transforms.Scale([224,224]),
                transforms.RandomHorizontalFlip(),
                transforms.TenCrop(64),
                
                transforms.Lambda(lambda crops: torch.stack([normalize(toTensor(crop)) for crop in crops])) 
                #transforms.ToTensor(),
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        else :
            print ("Please specify correct augmentation strategy : %s not defined"%(self.strategy));
            exit();
            
        return data_transforms;

