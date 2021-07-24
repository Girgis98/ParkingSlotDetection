
from Data.dataset import MarkingPoint
import codecs
import torch
import torchvision
import numpy as np
import cv2
import json
import os
import os.path
from torchvision.transforms import ToTensor
import math
from skimage import io, transform
from shutil import copyfile
from pathlib import Path
import random



# Data Augmenation
## done by rotating the image by a specified angle



def rotate_vector(vector, angle_degree):
    '''

    :param vector:
    :type vector:
    :param angle_degree:
    :type angle_degree:
    :return:
    :rtype:
    '''
    """Rotate a vector with given angle in degree."""
    angle_rad = math.pi * angle_degree / 180
    xval = vector[0]*math.cos(angle_rad) + vector[1]*math.sin(angle_rad)
    yval = -vector[0]*math.sin(angle_rad) + vector[1]*math.cos(angle_rad)
    coord_ls = [xval,yval]
    return coord_ls

def rotate_centralized_marks(centralied_marks, angle_degree):
    '''
    :param centralied_marks:
    :type centralied_marks:
    :param angle_degree:
    :type angle_degree:
    :return:
    :rtype:
    '''
    """Rotate centralized marks with given angle in degree."""
    rotated_marks = []
    rotated_mark = []
    for i in range(len(centralied_marks)):
        mark = centralied_marks[i]
        mark_ls = [mark[0]-256,mark[1]-256,mark[2]-256,mark[3]-256,mark[4]-256]
        x_y = rotate_vector(mark_ls[0:2], angle_degree)
        dir_x_y = rotate_vector(mark_ls[2:4], angle_degree)
        x_y[0] += 256
        x_y[1] += 256
        dir_x_y[0] += 256
        dir_x_y[1] += 256

        if x_y[0]<10 or x_y[0]>502 or x_y[1]<10 or x_y[1]>502:
          pass
        elif dir_x_y[0]<0 or dir_x_y[0]>512 or dir_x_y[1]<0 or dir_x_y[1]>512:

          if dir_x_y[0]<0:
            offset = abs(dir_x_y[0])
            dir_x_y[0] += (offset + 2)

            if dir_x_y[1]>x_y[1]:
              dir_x_y[1] -= offset
    
            else:
              dir_x_y[1] += offset
              

          elif dir_x_y[0]>512:
            offset = abs(dir_x_y[0] - 512)
            dir_x_y[0] -= (offset + 2)

            if dir_x_y[1]>x_y[1]:
              dir_x_y[1] -= offset
            else:
              dir_x_y[1] += offset
               

          elif dir_x_y[1]<0:
            offset = abs(dir_x_y[1])
            dir_x_y[1] += (offset + 2)

            if dir_x_y[0]>x_y[0]:
              dir_x_y[0] -= offset
            else:
              dir_x_y[0] += offset
              

          elif dir_x_y[1]>512:
            offset = abs(dir_x_y[1] - 512)
            dir_x_y[1] -= (offset + 2)

            if dir_x_y[0]>x_y[0]:
              dir_x_y[0] -= offset
            else:
              dir_x_y[0] += offset
              

          rotated_mark.append(x_y[0])
          rotated_mark.append(x_y[1])
          rotated_mark.append(dir_x_y[0])
          rotated_mark.append(dir_x_y[1])
          rotated_mark.append(mark_ls[4])
          rotated_marks.append(rotated_mark)
          rotated_mark = []  

        else:  
          rotated_mark.append(x_y[0])
          rotated_mark.append(x_y[1])
          rotated_mark.append(dir_x_y[0])
          rotated_mark.append(dir_x_y[1])
          rotated_mark.append(mark_ls[4])
          rotated_marks.append(rotated_mark)
          rotated_mark = []
    return rotated_marks

def rotate_image(image, angle_degree):
    '''

    :param image:
    :type image:
    :param angle_degree:
    :type angle_degree:
    :return:
    :rtype:
    '''
    """Rotate image with given angle in degree."""
    rows, cols, _ = image.shape
    #print(rows,cols)
    rotation_matrix = cv2.getRotationMatrix2D((rows/2, cols/2), angle_degree, 1)
    return cv2.warpAffine(image, rotation_matrix, (rows, cols))

def generate_dataset(parking_image,name,angle_step,dst_path):
    '''

    :param parking_image:
    :type parking_image:
    :param name:
    :type name:
    :param angle_step:
    :type angle_step:
    :param dst_path:
    :type dst_path:
    :return:
    :rtype:
    '''

    for angle in range(0, 360, angle_step):
        added_name= str(angle)
        rotated_marks = rotate_centralized_marks(parking_image['marks'], angle)
        rotated_image = rotate_image(np.array(parking_image['image'].permute(1,2,0)), angle)
        marking_points = []
        for label in np.array(rotated_marks):
            marking_points.append(MarkingPoint(*label))
        if len(marking_points)!=0:
          in_image = rotated_image
          in_image = cv2.cvtColor(np.array(in_image), cv2.COLOR_RGB2BGR)
          in_image = cv2.convertScaleAbs(in_image, alpha= 255.0)
          cv2.imwrite(f"{dst_path}/{name}_{added_name}.jpg", in_image)
          in_json ={}
          in_json = {'marks':marking_points,'slots':parking_image['slots']}
          path = f"{dst_path}/{name}_{added_name}.json"
          json.dump(in_json, codecs.open(path, 'w', encoding='utf-8'),
                    separators=(',', ':'), sort_keys=True) # this saves the array in .json format
    return 0

def load_jpg_json(path):
    '''

    :param path:
    :type path:
    :return:
    :rtype:
    '''
    try:
        jpg_names = []
        json_names =[]
        for file in os.listdir(path):
                  if file.endswith(".jpg"):
                      jpg_names.append(os.path.splitext(file)[0])
                  if file.endswith(".json"):
                      json_names.append(os.path.splitext(file)[0])
        return jpg_names,json_names
    except:
        return load_jpg_json(path)

def check_jpeg_json(json_names,jpg_names):
    '''

    :param json_names:
    :type json_names:
    :param jpg_names:
    :type jpg_names:
    :return:
    :rtype:
    '''
    missing_json = []
    missing_jpg = []
    for i in range(len(json_names)):
        if jpg_names.count(json_names[i]) == 0:
          missing_json.append(jpg_names[i])

    for i in range(len(jpg_names)):
        if json_names.count(jpg_names[i]) == 0:
          missing_jpg.append(json_names[i])

    return   missing_json,missing_jpg

def remove_empty_images(names_jpg,root_dst):
    '''

    :param names_jpg:
    :type names_jpg:
    :param root_dst:
    :type root_dst:
    :return:
    :rtype:
    '''
    empty_file = []
    for i in range(len(names_jpg)):
        size = Path(f"{root_dst}/{names_jpg[i]}.jpg").stat().st_size
    if size == 0:
        empty_file.append(names_jpg[i])
    for i in range(len(empty_file)):
        os.remove(f"{root_dst}/{empty_file[i]}.json")
        os.remove(f"{root_dst}/{empty_file[i]}.jpg")


def checkIfDuplicates_2(listOfElems):
    '''

    :param listOfElems:
    :type listOfElems:
    :return:
    :rtype:
    '''
    ''' Check if given list contains any duplicates '''    
    setOfElems = set()
    for elem in listOfElems:
        if elem in setOfElems:
            return True
        else:
            setOfElems.add(elem)         
    return False

def remove_miss_labeled_image(names,root_dst):
    '''

    :param names:
    :type names:
    :param root_dst:
    :type root_dst:
    :return:
    :rtype:
    '''
    empty = []
    for i, name in enumerate(names):
        with open(f"{root_dst}/{name}.json", ) as file:
          a = json.load(file)['marks']
          if len(a[0])<5:
            empty.append(name)
    for i in range(len(empty)):
        os.remove(f"{root_dst}/{empty[i]}.json")
        os.remove(f"{root_dst}/{empty[i]}.jpg")


def invert_image(parking_image,name,dst_path):
  '''

  :param parking_image:
  :type parking_image:
  :param name:
  :type name:
  :param dst_path:
  :type dst_path:
  :return:
  :rtype:
  '''
  invert = torchvision.transforms.RandomVerticalFlip(p=1)  #RandomHorizontalFlip(p=1)
  image_inverted = invert(parking_image['image'])
  image = parking_image['image']
  label = parking_image['marks']
  label_inverted_full = []
  label_inverted = []
  marking_points = []
  for i in range (len(label)):
    label_inverted.append (label[i][0])
    label_inverted.append (512 - label[i][1])
    label_inverted.append (label[i][2])
    label_inverted.append (512 - label[i][3])
    label_inverted.append (label[i][4])
    label_inverted_full.append(label_inverted)
    label_inverted = []
  for label in np.array(label_inverted_full):  
      marking_points.append(MarkingPoint(*label))  

  in_image = image_inverted.permute(1, 2, 0)
  in_image = cv2.cvtColor(np.array(in_image), cv2.COLOR_RGB2BGR)
  in_image = cv2.convertScaleAbs(in_image, alpha=(255.0))
  cv2.imwrite((f"{dst_path}/{name}.jpg"), in_image)
  in_json ={}
  in_json = {'marks':marking_points,'slots':parking_image['slots']}
  path = f""{dst_path}/{name}.json"
  json.dump(in_json, codecs.open(path, 'w', encoding='utf-8'),
            separators=(',', ':'), sort_keys=True) # this saves the array in .json format

"""#### Converting jpg to pt"""

def jpg_2_pt(root_src,names):
    '''

    :param root_src:
    :type root_src:
    :param names:
    :type names:
    :return:
    :rtype:
    '''
    for i in range(len(names)):
        image = cv2.imread(f"{root_src}/{names[i]}.jpg")
        t = torch.from_numpy(image)
        torch.save(t,f"{root_src}/{names[i]}.pt")

"""Generate shuffled data """

def random_generation(file_size,root_src,root_dst,added_name):
  '''

  :param file_size:
  :type file_size:
  :param root_src:
  :type root_src:
  :param root_dst:
  :type root_dst:
  :param added_name:
  :type added_name:
  :return:
  :rtype:
  '''
  names = []
  for file in os.listdir(root_src):
              if file.endswith(".json"):
                  names.append(os.path.splitext(file)[0])

  random_list = []
  while len(random_list)<file_size:
    n = random.randint(0,len(names))
    if not n in random_list:
      random_list.append(n)

  for i in range(0,len(random_list)):
      src_json = f"{root_src}/{names[i]}.json"
      dst_json = f"{root_dst}/{added_name}{names[i]}.json"
      copyfile(src_json, dst_json)
      src_jpg = f"{root_src}/{names[i]}.jpg"
      dst_jpg = f"{root_dst}/{added_name}{names[i]}.jpg"
      copyfile(src_jpg, dst_jpg)



"""## Rescale Class"""

class Rescale(object):
    """Rescale the image in a sample to a given size.
    """

    def __init__(self, output_size):
        '''

        :param output_size:
        :type output_size:
        '''
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        '''

        :param sample:
        :type sample:
        :return:
        :rtype:
        '''
        image =sample['image']
        marking_points = sample['marks']
        slots = sample['slots']

        h, w = image.shape[1:3]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        
        img = image.permute(1, 2, 0)
        img = transform.resize(img, (new_h, new_w))
        to_Tensor = ToTensor()
        img = to_Tensor(img)
        
        mult = ([(new_h / h) ,(new_w / w) ,(new_h / h) ,(new_w / w)])
        for i in range(len(marking_points)):
            iterable_mp = list(marking_points[i][0:4])
            for j in range(4):
                iterable_mp[j] = (iterable_mp[j] * mult[j])
            iterable_mp.append(marking_points[i][-1])    
            marking_points[i] = MarkingPoint(*iterable_mp)    
        return {'image': img,'marks': marking_points,'slots': slots}