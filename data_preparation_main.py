import os.path
from shutil import copyfile
from Data.preprocessing import load_jpg_json, generate_dataset, random_generation, invert_image, check_jpeg_json, \
    remove_empty_images, remove_miss_labeled_image, Rescale
from Data.dataset import load

root_src = r"source_path"
root_dst = r"destination_path"

"""### Load json files and jpeg files and check that their number is the same

"""

jpg_names = []
json_names = []
jpg_names, json_names = load_jpg_json(root_src)
print(len(json_names), len(jpg_names))

angle = 10

"""# Generate images in 30 seprate folders
"""

park_dataset = load(root_src)
len(park_dataset.sample_names)

parking_image0 = park_dataset[0]
parking_image0['marks']

for i in range(0, 30):
    for j in range(i * 500, (i * 500) + 500):
        name = f'{i}/{park_dataset.sample_names[j]}_aug'
        generate_dataset(park_dataset[j], name, angle, root_dst)

"""## Generate shuffled dataset from given dataset """

random_generation(20, root_src, root_dst)

"""## Reflect images"""

for i in range(0, len(park_dataset)):
    name = park_dataset.sample_names[i] + "x_ref"
    invert_image(park_dataset[i], name, root_dst)

"""##Copy specific files"""

names = []
added_name = ""
for file in os.listdir(root_src):
    if file.endswith(".json"):
        names.append(os.path.splitext(file)[0])
for i in range(0, len(names)):
    src_json = f"{root_src}/{names[i]}.json"
    dst_json = f"{root_dst}/{added_name}{names[i]}.json"
    copyfile(src_json, dst_json)
    src_jpg = f"{root_src}/{names[i]}.jpg"
    dst_jpg = f"{root_dst}/{added_name}{names[i]}.jpg"
    copyfile(src_jpg, dst_jpg)

"""# Check missed files"""

missing_json, missing_jpg = check_jpeg_json(json_names, jpg_names)
for i in range(len(missing_json)):
    print(missing_json[i])
for i in range(len(missing_jpg)):
    print(missing_jpg[i])

for i in range(len(jpg_names)):
    remove_empty_images(jpg_names[i])
    remove_miss_labeled_image(jpg_names[i])

"""# Rescalling the images """

image_rescaler = Rescale((512, 512))
for i in range(len(park_dataset)):
    if park_dataset[i]['image'].shape[1] != 512:
        temp = image_rescaler(park_dataset[i])
        park_dataset.__set__(i, temp)
