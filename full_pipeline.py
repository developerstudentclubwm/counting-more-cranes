import glob
import os
import sys
import shutil
import json
import csv
import time
import argparse
from datetime import date
import gc
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
from tile_mosaics import tile_mosaic, get_tile_dir

sys.path.append(os.path.join(os.getcwd(), 'density_estimation', 'ASPDNet'))
sys.path.append(os.path.join(os.getcwd(), 'object_detection'))

from utils import *
from density_estimation.ASPDNet_model import ASPDNetLightning
from density_estimation.ASPDNet.model import ASPDNet
from object_detection.faster_rcnn_model import *

def run_pipeline(mosaic_fp, model_name, model_save_fp, write_results_fp, num_workers, model_hyperparams = None, save_preds = False, use_cpu = False, batch_size = 32):

    """
    A wrapper function that assembles all pipeline elements.
    This function predicts a total count for a given mosaic (i.e., a flight line).
    Inputs:
     - mosaic_fp: a text file with mosaics to predict a total count for
     - model_name: the model to use for the prediction component... currently, should be one of faster_rcnn or ASPDNet
     - model_save_fp: the saved model as either a .pth or .ckpt file
     - write_results_fp: the CSV file to write the run results to
     - num_workers: the number of workers to use for the tile dataloader
     - model_hyperparams: any hyperparameters to use for the model
     - save_preds: whether or not to save visualized tile predictions
     - use_cpu: whether or not to explicitly use CPU for prediction
     - batch_size: the batch size used for prediction
    Outputs:
     - A total count for the input mosaic (also saves run results to desired CSV file)
    """

    start_time = time.time()

    #Loading in multiple images in one file 
    tile_dir = get_tile_dir()
    with open(mosaic_fp, 'r') as f:
        file_paths = f.readlines()
    
    file_paths = [path.strip() for path in file_paths]

    #Store filepaths to mosaics in text file
    for i, path in enumerate(file_paths):
        #PREDICT ON TILES:
        tile_dataset = BirdDatasetPREDICTION(path, model_name)
        tile_dataloader = DataLoader(tile_dataset, batch_size = batch_size, shuffle = False, 
                                     collate_fn = collate_tiles_PREDICTION, num_workers = num_workers)
        print(f'\nPredicting on {len(tile_dataset)} tiles...')
        
        #  get device, only if use_cpu isn't explicitly specified
        if use_cpu:
            device = 'cpu'
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using {device} for prediction...')

        #  grabbing any constructor hyperparams - currently, only necessary for our Faster R-CNN impelementation!
        if model_hyperparams is not None:
            constructor_hyperparams = model_hyperparams['constructor_hyperparams']

        #  loading the model from either a PyTorch Lightning checkpoint or a PyTorch model save
        print(f'\tLoading the saved {model_name} model...')
        if model_name == 'faster_rcnn':
            if model_save_fp.endswith('.pth'):
                model = get_faster_rcnn(backbone = 'ResNet50', num_classes = 2, **constructor_hyperparams).to(device) #making sure to pass in the constructor hyperparams here
                model.load_state_dict(torch.load(model_save_fp))
                pl_model = FasterRCNNLightning(model)
            elif model_save_fp.endswith('.ckpt'):
                model = get_faster_rcnn(backbone = 'ResNet50', num_classes = 2, **constructor_hyperparams).to(device)
                pl_model = FasterRCNNLightning.load_from_checkpoint(model_save_fp, model = model)
            else:
                raise NameError('File is not of type .pth or .ckpt')
        elif model_name == 'ASPDNet':
            if model_save_fp.endswith('.pth'):
                model = ASPDNet(allow_neg_densities = False).to(device)
                model.load_state_dict(torch.load(model_save_fp)) 
                pl_model = ASPDNetLightning(model)
            elif model_save_fp.endswith('.ckpt'):
                model = ASPDNet(allow_neg_densities = False).to(device)
                pl_model = ASPDNetLightning.load_from_checkpoint(model_save_fp, model = model)
            else:
                raise NameError('File is not of type .pth or .ckpt')
        else:
            raise NameError(f'Model "{model_name}" is not a supported model type')

        print('\tProducing counts...')

        if save_preds and not os.path.exists('mosaic_tiles/predictions'): #create an empty directory for preds
            os.mkdir('mosaic_tiles/predictions')

        pred_start_time = time.time()

        total_count = 0
        pl_model.model.eval() #making sure we're in eval mode...
        for i, batch in enumerate(tile_dataloader):

            print(f'\t\tBatch {i + 1}/{len(tile_dataloader)}')
            tile_batch, tile_fps = batch #getting out the content from the dataloader
            tile_batch = tile_batch.to(device) #loading the batch onto the same device as the model

            if model_name == 'faster_rcnn':
                tile_batch = list(tile_batch) #turning it into a list of tensors, as required by Faster R-CNN

            with torch.no_grad(): #disabling gradient calculations... not necessary, since we're just doing forward passes!
                tile_preds = pl_model(tile_batch) #getting predictions... not yet counts!

            if model_name == 'faster_rcnn': #predicting on the tiles and extracting counts for each tile
                tile_counts = [len(p['boxes'].tolist()) for p in tile_preds]
                total_count += sum(tile_counts) #adding in the counts for this batch of tiles
            elif model_name == 'ASPDNet':
                total_count += float(tile_preds.sum()) #adding in the counts

            #Saving predictions as we go
            if save_preds:
                if model_name == 'faster_rcnn': #saving tiles w/bboxes overlaid
                    for i, (img, fp) in enumerate(zip(tile_batch, tile_fps)):
                        img = img.cpu() #moving to CPU to avoid CUDA errors...
                        img = (np.moveaxis(img.numpy(), 0, -1) * 255).astype(np.uint8)
                        pred_boxes = tile_preds[i]['boxes'].tolist()

                        offset = 15
                        background = Image.new('RGB', (img.shape[0], img.shape[1] + offset), (255, 255, 255))
                        pil_img = Image.fromarray(img)
                        Image.Image.paste(background, pil_img, (0, offset))

                        draw = ImageDraw.Draw(background)
                        for b in pred_boxes: #drawing bboxes onto the tile
                            b = (b[0], b[1] + offset, b[2], b[3] + offset)
                            draw.rectangle(b, outline = 'red', width = 1)
                        draw.text((2, 2), str(len(pred_boxes)), fill = (0, 0, 0))

                        tile_save_name = f'{os.path.basename(fp).split(".")[0]}_FRCNN.png'

                        background.save(os.path.join('mosaic_tiles', 'predictions', tile_save_name))
                elif model_name == 'ASPDNet': #saving the pred densities for each tile
                    cm = plt.get_cmap('jet')
                    for den, fp in zip(list(tile_preds), tile_fps):
                        den = den.cpu()
                        colored_image = cm(den.numpy()) #applying the color map... makes it easier to look at!

                        # plotting the density w/the predicted count
                        fig, ax = plt.subplots()

                        ax.imshow(colored_image)
                        ax.set_title(round(float(den.sum()), 3))
                        ax.axis('off')
                        ax.set_xticks([])
                        ax.set_yticks([])

                        tile_save_name = f'{os.path.basename(fp).split(".")[0]}_ASPDNET.png'

                        fig.savefig(os.path.join('mosaic_tiles', 'predictions', tile_save_name), dpi = 50, bbox_inches = 'tight')
                        plt.close(fig)

        pred_time = time.time() - pred_start_time

        if save_preds:
            print(f'Predictions saved at {os.path.join("mosaic_tiles", "predictions")}')
        print('Done with prediction!')

        #SAVING/RETURNING RESULTS:
        fields = ['date', 'time', 'mosaic_fp', 'num_tiles', 'total_count', 'model', 'total_run_time', 'prediction_run_time']
        curr_time = str(time.strftime('%H:%M:%S', time.localtime()))
        curr_date = str(date.today())
        pipeline_time = time.time() - start_time
        new_row = [curr_date, curr_time, path, len(tile_dataset), int(total_count), model_name, pipeline_time, pred_time] #all of the run results to include
    
        if not os.path.isfile(write_results_fp): #either creating a new results CSV or adding to the existing file
                with open(write_results_fp, 'w') as file:
                    csvwriter = csv.writer(file)
                    csvwriter.writerow(fields)
                    csvwriter.writerow(new_row)
        else:
                with open(write_results_fp, 'a') as file:
                    csvwriter = csv.writer(file)
                    csvwriter.writerow(new_row)
        print('\nResults saved!')

    return total_count

class BirdDatasetPREDICTION(Dataset):

    """
    A reduced version of BirdDataset to help read in and preprocess mosaic tiles for prediction.
    Inputs:
     - mosaic_fp: the filepath of the full mosaic, assume it exists
     - model_name: either Faster R-CNN or ASPDNet
    """

    def __init__(self, mosaic_fp, model_name):
        self.mosaic_fp = mosaic_fp
        tile_dir = get_tile_dir()
        self.tile_fps = []
        #Store filepaths to mosaics in text file
            
        # takes out everything before the filename as well as the file ending (i.e. "/path/to/file/img_name.tif" becomes "img_name")
        img_name = ''.join(mosaic_fp.split(os.path.sep)[-1].split('.')[:-1])
        # search for cached tiles
        tiles = glob.glob(f'./{tile_dir}/{img_name}-tile*')

        if len(tiles) == 0:
            tile_mosaic(mosaic_fp)
            tiles = glob.glob(f'./{tile_dir}/{img_name}-tile*')
        self.tile_fps = sorted(tiles)

        self.transforms = []
        if model_name == 'ASPDNet': #PyTorch's Faster R-CNN implementation handles normalization...
            self.transforms.append(A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], max_pixel_value = 1))
        self.transforms.append(ToTensorV2())
        self.transforms = A.Compose(self.transforms)

    def __getitem__(self, index):
        tile_fp = self.tile_fps[index]
        tile = Image.open(tile_fp).convert('RGB')
        tile = np.array(tile)

        preprocessed_tile = tile / 255
        preprocessed_tile = self.transforms(image = preprocessed_tile)['image'].float() #making sure that its dtype is float32

        return preprocessed_tile, tile_fp

    def __len__(self):
        return len(self.tile_fps)

def collate_tiles_PREDICTION(batch):
    """
    A workaround to ensure that we can retrieve the tile number for saving pipeline predictions.
    Inputs:
      - batch: a list of tuples w/format [(tile, tile_fp), ...]
    Outputs:
      - A tuple w/a list of tiles and a list of tile numbers
    """
    tiles = torch.stack([b[0] for b in batch]) 
    tile_fps = [b[1] for b in batch]

    return tiles, tile_fps

def str2bool(arg):
    """
    A simple workaround to ensure that bools from the argument parser are handled correctly.
    From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse.
    Inputs:
      - arg: the string argument passed to the parser
    Outputs:
      - True or False, depending on the inputted string
    """
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser() #an argument parser to collect arguments from the user

    #  required args
    parser.add_argument('mosaic_fp', help = 'text file with list of mosaics to process')
    
    parser.add_argument('model_name', help = 'the model name; either ASPDNet or faster_rcnn')
    parser.add_argument('model_fp', help = 'file path for model save; .ckpt or .pth')
    parser.add_argument('write_results_fp', help = 'file path to write pipeline run results to')

    #  optional args
    parser.add_argument('-bs', '--batch_size', help = 'the batch size for prediction', type = int, default = 32)
    parser.add_argument('-nw', '--num_workers', help = 'the number of workers to use in the tile dataloader', type = int, default = 0)
    parser.add_argument('-cfp', '--config_fp', help = 'file path for config, containing hyperparameters for model', default = None)
    parser.add_argument('-sp', '--save_preds', help = 'save predictions for tiles?', type = str2bool, default = False)
    parser.add_argument('-cpu', '--use_cpu', help = 'force use of cpu?', type = str2bool, default = False)

    args = parser.parse_args()

    if args.config_fp is not None: #getting the config file
        config = json.load(open(args.config_fp, 'r'))
        model_hyperparams = config[args.model_name + '_params'] #see config.json for structure of config file
    else:
        model_hyperparams = None

    run_pipeline(args.mosaic_fp, args.model_name, args.model_fp, 
                 args.write_results_fp, args.num_workers, model_hyperparams = model_hyperparams, 
                 save_preds = args.save_preds, use_cpu = args.use_cpu, batch_size = args.batch_size)
