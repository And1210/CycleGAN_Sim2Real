import argparse
from datasets import create_dataset
from utils import parse_configuration
from models import create_model
import os
from utils.visualizer import Visualizer
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

"""Performs validation of a specified model.

Input params:
    config_file: Either a string with the path to the JSON
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and
        model-specific settings.
"""
def validate(configuration):

    print('Initializing dataset...')
    val_dataset = create_dataset(configuration['val_dataset_params'])
    val_dataset_size = len(val_dataset)
    print('The number of validation samples = {0}'.format(val_dataset_size))

    print('Initializing model...')
    model = create_model(configuration['model_params'])
    model.setup()
    model.eval()

    print('Initializing visualization...')
    visualizer = Visualizer(configuration['visualization_params_validation'])   # create a visualizer that displays images and plots

    model.pre_epoch_callback(configuration['model_params']['load_checkpoint'])

    #Loops through all validation data and runs though model
    bs = configuration['val_dataset_params']['loader_params']['batch_size']
    for i, data in tqdm(enumerate(val_dataset)):
        model.set_input(data)  # unpack data from data loader
        # print(model.input.shape)
        model.test()           # run inference
        for j in range(bs):
            # img = model.input[j].permute(1, 2, 0).cpu().detach().numpy()
            # trg = model.target[j].permute(1, 2, 0).cpu().detach().numpy()
            # out_img = model.output[j].permute(1, 2, 0).cpu().detach().numpy()

            # if (not os.path.exists(val_dataset.dataset.sim_img_paths[i*bs+j].replace('/val/', '/val_C6/'))):
            out_trg_img = model.fake_B[j].permute(1, 2, 0).cpu().detach().numpy()

            #Save images
            out_trg_img = out_trg_img * 255
            out_trg_img = out_trg_img.astype(np.uint8)
            output = Image.fromarray(out_trg_img)
            output.save(val_dataset.dataset.sim_img_paths[i*bs+j].replace('/images/', '/images_S2R_MK1/'))

    #Where results are calculated and visualized
    # model.post_epoch_callback(configuration['model_params']['load_checkpoint'], visualizer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model validation.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()

    print('Reading config file...')
    configuration = parse_configuration(args.configfile)
    if (configuration['model_params']['load_checkpoint'] == -2):
        for epoch in range(configuration['model_params']['epoch_list'][0], configuration['model_params']['epoch_list'][1]):
            configuration['model_params']['load_checkpoint'] = epoch
            validate(configuration)
    else:
        validate(configuration)
