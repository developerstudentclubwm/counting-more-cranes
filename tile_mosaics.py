import argparse
import os
import gc
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import empty, float32
from utils import purge_invalid_bboxes
from PIL import Image
import numpy as np

tiles_dir = 'mosaic_tiles'

# for use outside of this file
def get_tile_dir() -> str:
    return tiles_dir

def pad_array(array: np.array, tile_size = (200,200)) -> np.array:
    tw, th = tile_size
    width, height = array.shape
    return np.pad(array, ((0, tw - width % tw), (0, th - height % th)))

def tile_and_annotate_file(file, bboxes, labels, normalize=False, tile_size=(200,200)):
    pass

def annotate_cached_tiles(file, bboxes, labels, normalize=False):
    img_name = file.split(os.path.sep)[-1]
    tiles = glob.glob(os.path.join(tiles_dir, f'tile_*-{img_name}'))

    img_array = np.asarray(Image.open(file).convert('L'))
    iw = img_array.width + (tw - img_array.width % tw)

    normalization = A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], max_pixel_value = 1)

    tile_tensors = []
    targets = []
    for tile in tiles:
        tile_array = np.asarray(Image.open(tile).convert('L'))
        tw, th = tile_array.shape
        num = int(tile[tile.index('tile_')+len('tile_'):tile.index('-')])
        # Convert 
        y = (num // (iw//tw)) * th
        x = (num % (iw//tw)) * tw
        bboxes = [[xmin-x, ymin-y, xmax-x, ymax-y] for xmin, ymin, xmax, ymax in bboxes]
        
        transforms = A.Compose([], A.BboxParams(format= 'pascal_voc', label_fields = ['class_labels'], min_visibility = 0.5))
        t = transforms(image = tile_array, bboxes = bboxes, class_labels = labels)
        if len(t['bboxes']) == 0:
            new_bboxes = empty((0, 4), dtype = float32)
        else:
            new_bboxes = purge_invalid_bboxes(t['bboxes'])
        target_dict = {'boxes' : new_bboxes, 'labels' : np.ones((len(new_bboxes), ))}

        image = t['image'] / 255
        if normalize:
            image = normalization(image = image)['image']
        tile_tensors.append(ToTensorV2()(image = image)['image'])
        targets.append(target_dict)

    return tiles, targets

def tile_file(file: str, tile_size = (200,200)):
    tw, th = tile_size
    img_name = file.split(os.path.sep)[-1]
    # open image and store as numpy array
    img = Image.open(file).convert('L')
    img_array = np.asarray(img)
    # pad image to be divisible by tw and th
    padded_array = pad_array(img_array, tile_size)
    # free up some memory
    del img, img_array
    gc.collect()

    tile_count = 0
    for row in range(0, padded_array.shape[0]-1, tw):
        for col in range(0, padded_array.shape[1]-1, th):
            # file name of new tile
            tile_name = os.path.join(tiles_dir, f"tile_{tile_count}-{img_name}")
            # if tile already exists, skip
            if not os.path.isfile(tile_name):
                # segment padded array into a tw by th tile
                tile_array = padded_array[row:row+tw][:,col:col+th]
                # if tile is all zeroes, skip
                if tile_array.any():
                    tile_img = Image.fromarray(tile_array, 'L')
                    tile_img.save(tile_name)
                    tile_count += 1

def tile_mosaic(file: str, tile_size = (200,200)):
    """
    file is the filepath to the mosaic
    tile_size is the size of the tiles you want to create

    If the size of the image at file does not fit tile_size, it will be padded with 0 values
    """
    if not file.endswith('.tif') and file.endswith('.TIF'):
        raise ValueError(f'File {file} is not a .tif or .TIF file')
    # make tile directory if it doesn't exist
    if not os.path.exists(tiles_dir):
        os.mkdir(tiles_dir)
    tile_file(file, tile_size)

def tile_mosaics_dir(dir: str, tile_size = (200, 200)):
    """
    dir is the name of the directory containing the mosaic files

    This function will walk through dir and make tiles out of any .tif files it finds.
    """
    # make tile directory if it doesn't exist
    if not os.path.exists(tiles_dir):
        os.mkdir(tiles_dir)

    for (root, _, files) in os.walk(dir):
        for file in files:
            path = os.path.join(root, file)
            print(f"Tiling {file}...")
            tile_file(path, tile_size)
            
    print("Done tiling!")


def tile_mosaics_from_file(mosaic_file: str, tile_size = (200, 200)):
    """
    mosaic_file is the filepath of a file containing the filepaths of mosaic files separated by newline characters

    This function will tile any .tif or .TIF files it finds in mosaic_file
    """
    # make tile directory if it doesn't exist
    if not os.path.exists(tiles_dir):
        os.mkdir(tiles_dir)
    with open(mosaic_file, 'r') as f:
        lines = f.readlines()
        files = [path.strip() for path in lines]
    
    for file in files:
        if not file.endswith('.tif') and not file.endswith('.TIF'):
            continue
        print(file)
        tile_file(file, tile_size)
        
    print("Done tiling!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tw', '--tile-width', type=int, default=200, help='width of the tiles')
    parser.add_argument('-th', '--tile-height', type=int, default=200, help='height of the tiles')
    parser.add_argument('-f', '--file', type=str, help='filepath for mosaic', default=None)
    parser.add_argument('-d', '--directory', type=str, help='filepath for mosaic directory', default=None)
    parser.add_argument('-mf', '--mosaic-file', type=str, help='filepath for file of mosaics', default=None)
    parser.add_argument('-a', '--annotate', action='store_true', help='annotate images')         # padding 
    args = parser.parse_args()
    print(args.annotate)
    # if more than one of file, directory, or mosaic_file were passed
    if int(bool(args.file)) + int(bool(args.directory)) + int(bool(args.mosaic_file)) != 1:
        raise argparse.ArgumentTypeError('Include either \'--file\' or \'--directory\' or \'--mosaic-file\' in command line')

    tw, th = args.tile_width, args.tile_height
    # tile a single .tif file
    if args.file:
        tile_mosaic(args.file, (tw, th))
    # tile all the .tif files in a directory
    elif args.directory:
        tile_mosaics_dir(args.directory, (tw, th))
    # tile all .tif files contained in a file
    elif args.mosaic_file:
        tile_mosaics_from_file(args.mosaic_file, (tw, th))

    # Run this to test:
    # For mosaic: time python3 tile_mosaics.py -f [mosaic]
    # For mosaic file: time python3 tile_mosaics.py -mf [mosaic file] 
    # For directory: time python3 tile_mosaics.py -d [directory] 

if __name__ == '__main__':
    main()

