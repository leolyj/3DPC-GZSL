#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on ModelNet40 dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import os
import numpy as np

# Dataset
from torch.utils.data import DataLoader
from gzsl3d.kpconv.datasets.SemanticKitti import *
from gzsl3d.kpconv.utils.config import Config
from gzsl3d.kpconv.utils.tester import ModelTester
from gzsl3d.kpconv.models.architectures import KPFCNN


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

def model_choice(chosen_log):

    ###########################
    # Call the test initializer
    ###########################

    # Automatically retrieve the last trained model
    if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS']:

        # Dataset name
        test_dataset = '_'.join(chosen_log.split('_')[1:])

        # List all training logs
        logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])

        # Find the last log of asked dataset
        for log in logs[::-1]:
            log_config = Config()
            log_config.load(log)
            if log_config.dataset.startswith(test_dataset):
                chosen_log = log
                break

        if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS']:
            raise ValueError('No log of the dataset "' + test_dataset + '" found')

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    return chosen_log


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ###############################
    # Choose the model to visualize
    ###############################

    #   Here you can choose which model you want to test with the variable test_model. Here are the possible values :
    #
    #       > 'last_XXX': Automatically retrieve the last trained model on dataset XXX
    #       > '(old_)results/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model
    os.chdir("../../")
    chosen_log = 'results/Log_SemanticKITTI' # Config from complete training

    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = None
    chosen_chkp  = "../seg/run/sk/20_model.pth.tar"
    bias = 0.2
    weight = 50
    generative = True
    # Choose to test on validation or test split
    on_val = True
    num_votes = 500 # 500
    # Deal with 'last_XXXXXX' choices
    chosen_log = model_choice(chosen_log)

    class args_default():

        def __init__(self):
            self.load_embedding = "glove_w2v"
            self.embed_dim = 600

    args = args_default()  


    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    ###############
    # Previous chkp
    ###############

    # Find all checkpoints in the chosen training folder
    #Take the ckpt out of the run model 
    FLAG_ckpt_run = True 
    #chkp_path = os.path.join(chosen_log, 'checkpoints')
    #chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

    # Find which snapshot to restore
    #if chkp_idx is None:
    #    chosen_chkp = 'current_chkp.tar'
    #else:
    #    chosen_chkp = np.sort(chkps)[chkp_idx]
    #chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)

    # Initialize configuration class
    config = Config()
    config.load(chosen_log)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    #config.augment_noise = 0.0001
    #config.augment_symmetries = False
    config.batch_num = 8 #Zuerst ausgeklammert
    config.val_batch_num = 8
    #config.in_radius = 51
    config.validation_size = 400
    config.input_threads = 10

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    if on_val:
        set = 'validation'
    else:
        set = 'test'

    # Initiate dataset
    if config.dataset == 'ModelNet40':
        test_dataset = ModelNet40Dataset(config, train=False)
        test_sampler = ModelNet40Sampler(test_dataset)
        collate_fn = ModelNet40Collate
    elif config.dataset == 'S3DIS':
        test_dataset = S3DISDataset(config, set='validation', use_potentials=True)
        test_sampler = S3DISSampler(test_dataset)
        collate_fn = S3DISCollate
    elif config.dataset == 'SemanticKitti':
        
        test_dataset = SemanticKittiDataset(config, args, set=set, balance_classes=False, ignored_labels = np.sort([0]), visual_set=True)
        test_sampler = SemanticKittiSampler(test_dataset)
        collate_fn = SemanticKittiCollate
    else:
        raise ValueError('Unsupported dataset : ' + config.dataset)

    # Data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    
   
    net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)

    # Define a visualizer class
    tester = ModelTester(net, chkp_path=chosen_chkp, generative=generative)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart test')
    print('**********\n')
    # Training
    if config.dataset_task == 'classification':
        tester.classification_test(net, test_loader, config)
    elif config.dataset_task == 'cloud_segmentation':
        tester.cloud_segmentation_test(net, test_loader, config, num_votes = num_votes)
    elif config.dataset_task == 'slam_segmentation':
        #This is used
        tester.slam_segmentation_test(net, test_loader, config, num_votes = num_votes, bias=bias, weight= weight)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)
