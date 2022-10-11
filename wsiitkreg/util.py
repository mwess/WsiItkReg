# Functions

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import SimpleITK as sitk
import torchstain

# Util functions
####### Utility function #######
def command_iteration(method):
    if (method.GetOptimizerIteration() == 0):
        print("Estimated Scales: ", method.GetOptimizerScales())
    print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():7.5f} : {method.GetOptimizerPosition()}")

def plot_sitk_image(image):
    plt.imshow(sitk.GetArrayFromImage(image))

def pil_to_sitk(image):
    # First convert to grayscale
    grayscaled_image = PIL.ImageOps.grayscale(image)
    # To numpy array
    np_image = np.array(grayscaled_image)
    # Rescale numeric values
    np_image = np_image/255.*65025.
    # Reset type
    np_image = np_image.astype(float)
    # Convert to sitk
    sitk_image = sitk.GetImageFromArray(np_image)
    # Convert pixeltype to float32
    sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)
    return sitk_image

def sitk_to_pil(image):
    # Convert to numpy array
    np_image = sitk.GetArrayFromImage(image)
    # Rescale
    np_image = np_image/65025.*255.
    # Convert to integer for PIL
    np_image = np_image.astype(np.uint8)
    return PIL.Image.fromarray(np_image)

def extract_image_from_slide(slide):
    return slide.read_region((0,0), 0 , slide.dimensions)  

# Normalization
def normalize(source_path, target_path):
    size = 1024
    src_img = cv2.resize(cv2.cvtColor(cv2.imread(source_path), cv2.COLOR_BGR2RGB), (size, size))
    target_img = cv2.resize(cv2.cvtColor(cv2.imread(target_path), cv2.COLOR_BGR2RGB), (size, size))
    cv2.imwrite('fixed1.tif', src_img)
    #target = cv2.resize(cv2.cvtColor(cv2.imread(source_path), cv2.COLOR_BGR2RGB), (size, size))
    #to_transform = cv2.resize(cv2.cvtColor(cv2.imread(target_path), cv2.COLOR_BGR2RGB), (size, size))
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='numpy')
    normalizer.fit(src_img)

    # t_ = time.time()
    norm, H, E = normalizer.normalize(I=target_img, stains=True)
    cv2.imwrite('moving1.tif', norm)

# Functions for processing validation data

FROZEN_STORAGE_DATA_ORDER = {
    'MS-F-1': 0,
    'MS-V-1': 1,
    'MS-FV-1': 2,
    'MS-F-2': 3,
    'MS-V-2': 4,
    'MS-FV-2': 5,
    'MS-F-3': 6,
    'MS-V-3': 7,
    'MS-FV-3': 8,
    'MM-F-1': 9,
    'MM-V-1': 10,
    'MM-FV-1': 11,
    'MM-F-2': 12,
    'MM-V-2': 13,
    'MM-FV-2': 14,
    'MM-F-3': 15,
    'MM-V-3': 16,
    'MM-FV-3': 17,
}

def is_integerable(character):
    try:
        int(character)
        return True
    except ValueError:
        return False

def split_by_core_id_and_sort(file_list, key_list=None):
    if key_list is None:
        key_list=['A', 'B', 'C', 'D']
    dct = {}
    for file_ in file_list:
        # The filenames are currently of the form MM-FV-1_Overview_1_B.tif
        # The ids are in the last part of the filename: ....1_B.tif. The digit points towards the original position 
        # on the slide. The letter groups the cores together based on how they should have been ordered.
        id_ = file_.rsplit('_', maxsplit=1)[-1].split('.', maxsplit=1)[0]
        if id_ not in key_list:
            print(f'Unexpected id {id_} found. Ignoring...')
            continue
        if id_ not in dct:
            dct[id_] = []
        dct[id_].append(file_)
    for key in dct:
        dct[key] = sort_list(dct[key])
    return dct

def get_prefix(scan_name):
    return scan_name.split('_', maxsplit=1)[0]

def sort_list(lst):
    return sorted(lst, key=lambda key: FROZEN_STORAGE_DATA_ORDER[get_prefix(key)])

def split_training_pairs(dct):
    pairs = []
    for key in dct:
        core_slides = dct[key]
        for idx in range(len(core_slides)-1):
            pairs.append((core_slides[idx], core_slides[idx+1]))
    return pairs

def load_image_pair(source_image_path, target_image_path, use_cuda=False):
    batch = {'source_image_path': source_image_path, 'target_image_path': target_image_path}
    return batch

def ordered_images_to_batches(image_list):
    batches = []
    for i in range(len(image_list) - 1):
        batch = load_image_pair(image_list[i], image_list[i+1])
        batches.append(batch)
    return batches