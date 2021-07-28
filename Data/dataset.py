import numpy as np
import cv2
import json
import os
import os.path
import cv2 as cv
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from collections import namedtuple
import codecs
import random
import torch

"""
## Design of our input label:

*  Marking point tuple represents the coordinates of a point and coordinates of its shape and the shape of the slot line.
*   slot tuple reprents the two points, the angle between them and the shape of parking whether its parallel or perpendicular
"""

MarkingPoint = namedtuple('MarkingPoint', ['x', 'y', 'direction_x', 'direction_y', 'shape'])
Slot = namedtuple('Slot', ['p1', 'p2', 'angle', 'parking_shape'])

"""# Dataset class

*   initailization of images and its json files
*   getitem function: used for indexing dataset and return dictionary containing image and the marking point
* len function: return the length of available dataset
* set function: update the values in json file

### Class version that reads images from certain number of folders
"""

class ParkingSlotDataset(Dataset):
    """Parking slot dataset."""

    def __init__(self, root, names_ls = []):
        '''
        Initialize Dataset
        :param root: Path of the dataset folder
        :type root: string
        :param names_ls: list of files names 
        :type names_ls: list
        '''

        super(ParkingSlotDataset, self).__init__()
        self.root = root
        self.sample_names = names_ls.copy()
        self.test_names = []
        self.all_samples = []
        self.image_transform = ToTensor()
        self.mode = 'train'
        if len(names_ls) == 0:
            accum = 0
            sub_dirs_num = (len([f for f in os.listdir(root)]))
            print("Number of Subdirectories is: ", sub_dirs_num, "\n")
            for i in range(sub_dirs_num):
                sub_root = f'{root}/{i}'
                print('\nfile is:', i)
                for file in os.listdir(sub_root):
                    if file.endswith(".json"):
                        self.sample_names.append(f'{i}/{os.path.splitext(file)[0]}')
                print('file len:', len(self.sample_names) - accum)
                accum = len(self.sample_names)
                print('total len:', len(self.sample_names))

    def split(self):
        '''
        Splits dataset into train and test sets
        :return: returns 0 on success
        :rtype: int
        '''
        total_num = len(self.sample_names)
        train_num = int(0.85 * total_num)

        random.Random(7).shuffle(self.sample_names)

        self.all_samples = self.sample_names.copy()
        self.sample_names = self.all_samples[0: train_num]
        self.test_names = self.all_samples[train_num:]
        return 0

    def change_mode(self, mode):
        '''
        Change from/to test/train set
        :param mode: "train" or "test"
        :type mode: string
        :return: returns 0 on success
        :rtype: int
        '''
        if mode == 'train' and self.mode == 'test':
            self.sample_names, self.test_names = self.test_names.copy(), self.sample_names.copy()
            self.mode = 'train'

        elif mode == 'test' and self.mode == 'train':
            self.sample_names, self.test_names = self.test_names.copy(), self.sample_names.copy()
            self.mode = 'test'
        return 0    

    def __getitem__(self, index):
        '''
         Gets a sample by index
        :param index: index of sample
        :type index: int
        :return: sample
        :rtype: dictionary
        '''
        try:
            name = self.sample_names[index]
            image = cv.imread(f"{self.root}/{name}.jpg")
        except:
            print('error index is:', index)
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image2 = cv2.medianBlur(image, 9)
            smoothed = cv2.GaussianBlur(image2, (9, 9), 10)
            sharp = cv2.addWeighted(image2, 2, smoothed, -1, 0)
            bilat = cv2.bilateralFilter(sharp, 9, 75, 75)
            image = bilat

        except:
            print("error here", self.sample_names[index])
        image = self.image_transform(image)
        marking_points = []
        slots = []

        with open((f"{self.root}/{name}.json"), ) as file:
            if not isinstance(json.load(file)['marks'][0], list):
                with open((f"{self.root}/{name}.json"), ) as file:
                    marking_points.append(MarkingPoint(*json.load(file)['marks']))
            else:
                with open((f"{self.root}/{name}.json"), ) as file:
                    for label in np.array(json.load(file)['marks']):
                        marking_points.append(MarkingPoint(*label))

        with open((f"{self.root}/{name}.json"), ) as file:
            if (len(json.load(file)['slots']) == 0):
                pass
            else:
                with open((f"{self.root}/{name}.json"), ) as file:
                    if not isinstance(json.load(file)['slots'][0], list):
                        with open((f"{self.root}/{name}.json"), ) as file:
                            slots.append(Slot(*(json.load(file)['slots'])))
                    else:
                        with open((f"{self.root}/{name}.json"), ) as file:
                            for label in json.load(file)['slots']:
                                slots.append(Slot(*label))
        return {'image': image, 'marks': marking_points, 'slots': slots}

    def __len__(self):
        '''
        Return number of samples in working set
        :return: number of samples
        :rtype: int
        '''
        return len(self.sample_names)

    def __set__(self, index, value):
        '''
        Changes a sample by index
        :param index: sample index
        :type index: int
        :param value: new sample data
        :type value: dictionary
        :return: returns 0 on success
        :rtype: int
        '''
        # save file name then delete file then create another file with new data
        name = self.sample_names[index]
        in_image = value['image']
        in_image = in_image.permute(1, 2, 0)
        in_image = cv2.cvtColor(np.array(in_image), cv2.COLOR_RGB2BGR)
        in_image = cv.convertScaleAbs(in_image, alpha = 255.0)
        cv2.imwrite(f"{self.root}/{name}.jpg", in_image)
        in_json = {}
        in_json = {'marks': value['marks'], 'slots': value['slots']}
        path = f"{self.root}/{name}.json"
        json.dump(in_json, codecs.open(path, 'w', encoding = 'utf-8'),
                  separators = (',', ':'), sort_keys = True)  # this saves the array in .json format
        return 0


""" Class version that reads images from 1 folder (small scale)
"""


class ParkingSlotDatasetSingleFolder(Dataset):
    """Parking slot dataset."""

    def __init__(self, root, names_ls = []):
        '''
        Initialize Dataset
        :param root: Path of the dataset folder
        :type root: string
        :param names_ls: list of files names 
        :type names_ls: list
        '''
        super(ParkingSlotDatasetSingleFolder, self).__init__()
        self.root = root
        self.sample_names = names_ls.copy()
        self.image_transform = ToTensor()
        if len(names_ls) == 0:
            for file in os.listdir(root):
                if file.endswith(".json"):
                    self.sample_names.append(os.path.splitext(file)[0])

    def __getitem__(self, index):
        '''
         Gets a sample by index
        :param index: index of sample
        :type index: int
        :return: sample
        :rtype: dictionary
        '''
        try:
            name = self.sample_names[index]
            image = cv.imread(f"{self.root}/{name}.jpg")
        except:
            print('error index is:', index)
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image2 = cv2.medianBlur(image, 9)
            smoothed = cv2.GaussianBlur(image2, (9, 9), 10)
            sharp = cv2.addWeighted(image2, 2, smoothed, -1, 0)
            bilat = cv2.bilateralFilter(sharp, 9, 75, 75)
            image = bilat

        except:
            print("error here", self.sample_names[index])
        image = self.image_transform(image)
        marking_points = []
        slots = []

        with open(f"{self.root}/{name}.json", ) as file:
            if not isinstance(json.load(file)['marks'][0], list):
                with open(f"{self.root}/{name}.json", ) as file:
                    marking_points.append(MarkingPoint(*json.load(file)['marks']))
            else:
                with open(f"{self.root}/{name}.json", ) as file:
                    for label in np.array(json.load(file)['marks']):
                        marking_points.append(MarkingPoint(*label))

        with open(f"{self.root}/{name}.json", ) as file:
            if len(json.load(file)['slots']) == 0:
                pass
            else:
                with open(f"{self.root}/{name}.json", ) as file:
                    if not isinstance(json.load(file)['slots'][0], list):
                        with open(f"{self.root}/{name}.json", ) as file:
                            slots.append(Slot(*(json.load(file)['slots'])))
                    else:
                        with open(f"{self.root}/{name}.json", ) as file:
                            for label in json.load(file)['slots']:
                                slots.append(Slot(*label))
        return {'image': image, 'marks': marking_points, 'slots': slots}

    def __len__(self):
        '''
        Return number of samples in working set
        :return: number of samples
        :rtype: int
        '''
        return len(self.sample_names)

    def __set__(self, index, value):
        '''
        Changes a sample by index
        :param index: sample index
        :type index: int
        :param value: new sample data
        :type value: dictionary
        :return: returns 0 on success
        :rtype: int
        '''
        # save file name then delete file thrn create another file with new data
        name = self.sample_names[index]
        in_image = value['image']
        in_image = in_image.permute(1, 2, 0)
        in_image = cv2.cvtColor(np.array(in_image), cv2.COLOR_RGB2BGR)
        in_image = cv.convertScaleAbs(in_image, alpha = (255.0))
        cv2.imwrite((f"{self.root}/{name}.jpg"), in_image)
        in_json = {}
        in_json = {'marks': value['marks'], 'slots': value['slots']}
        path = f"{self.root}/{name}.json"
        json.dump(in_json, codecs.open(path, 'w', encoding = 'utf-8'),
                  separators = (',', ':'), sort_keys = True)  # this saves the array in .json format


"""## load function"""


def load(path, ls = []):
    '''
    A recursive function to load big datasets 
    :param path: Path of the dataset folder 
    :type path: string
    :param ls: List of samples names
    :type ls: List
    :return: dataset
    :rtype: ParkingSlotDataset
    '''
    try:
        park_dataset1 = ParkingSlotDataset(path, ls)
        return park_dataset1
    except:
        return load(path)


def collate_mod(data):
    '''
    Custom dataloader colate function
    :param data: List of samples
    :type data: List
    :return: batch of samples
    :rtype: dictionary
    '''
    lengths_marks = []
    lengths_slots = []
    for i in range(len(data)):
        l1 = len(data[i]['marks'])
        l2 = len(data[i]['slots'])
        lengths_marks.append(l1)
        lengths_slots.append(l2)

    max_len_marks = max(lengths_marks)
    max_len_slots = max(lengths_slots)

    features_marks = torch.zeros(len(data), max_len_marks, 5)
    features_slots = torch.zeros(len(data), max_len_slots, 4)

    dict_im = torch.empty(len(data), 3, 512, 512)
    for i in range(len(data)):
        j = len(data[i]['slots'])
        t = torch.zeros(max_len_slots - j, 4)
        features_slots[i] = torch.cat([(torch.Tensor(data[i]['slots'])), t])

        k = len(data[i]['marks'])
        t = torch.zeros(max_len_marks - k, 5)
        features_marks[i] = torch.cat([(torch.Tensor(data[i]['marks'])), t])

        dict_im[i] = data[i]['image']

    features_marks = features_marks.permute(1, 2, 0)
    features_slots = features_slots.permute(1, 2, 0)
    d = {'image': dict_im, 'marks': features_marks, 'slots': features_slots}

    return d
