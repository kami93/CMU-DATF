
import argparse
import importlib
import os
import sys 
from easydict import EasyDict
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

def parse_train_configs():
    parser = argparse.ArgumentParser(description='The Implementation using PyTorch')

    parser.add_argument('--config', type=str, default="config_simple", help="config file name")

    # Training Tag
    parser.add_argument('--tag', type=str, help="Add a tag to the saved folder")
    parser.add_argument('--exp_path', type=str, default='./experiment', help='Experient Directory')

    # Model type
    parser.add_argument('--model_name', type=str, default='SimpleEncDec', help="SimpleEncDec | SocialPooling | MATF | MATF_GAN | CAM | MACAM | R2P2_RNN | R2P2_SimpleRNN | Desire")
    
    # Hardware Parameters
    parser.add_argument('--num_workers', type=int, default=0, help="")
    parser.add_argument('--gpu_devices', type=str, default='0', help="GPU IDs for model running")

    # Dataset Parameters
    parser.add_argument('--dataset', type=str, default='argoverse', help="argoverse | nuscenes | carla")
    parser.add_argument('--train_cache', default=None, help="")
    parser.add_argument('--val_cache', default=None, help="")

    # Episode sampling parameters
    parser.add_argument('--sample_stride', type=int, default=1, help="Stride between reference frames in a single episode")

    # Trajectory Parameters
    parser.add_argument('--sampling_rate', type=int, default=2, help="Sampling Rate for Encoding/Decoding sequences") # Hz | 10 frames per sec % sampling_interval=5 => 2 Hz

    # Scene Context Parameters
    parser.add_argument('--map_version', type=str, default='2.0', help="Map version")
    ## Only used for MATFs
    parser.add_argument('--scene_dropout', type=float, default=0.5, help="")
    parser.add_argument('--scene_encoder', type=str, default='ShallowCNN', help="ShallowCNN | ResNet")
    parser.add_argument('--freeze_resnet', type=bool, default=True, help="")

    # Agent Encoding
    # (Not used for R2P2 and Desire)
    parser.add_argument('--agent_embed_dim', type=int, default=128, help="Agent Embedding Dimension")
    parser.add_argument('--lstm_layers', type=int, default=1, help="")
    parser.add_argument('--lstm_dropout', type=float, default=0.3, help="")

    # the number of candidate futures in generative models
    parser.add_argument('--num_candidates', type=int, default=12, help="Number of trajectory candidates sampled")
    
    # CSP Models
    parser.add_argument('--pooling_size', type=int, default=30, help="Map grid H and W dimension")

    # Attention Models
    parser.add_argument('--att_dropout', type=float, default=0.1, help="")

    # Normalizing Flow Models
    parser.add_argument('--multi_agent', type=int, default=1, help="Enables multi-agent setting for dataset")
    parser.add_argument('--beta', type=float, default=0.1, help="Ploss beta parameter")
    parser.add_argument('--velocity_const', type=float, default=0.5, help="Constant multiplied to dx in verlet integration")
    parser.add_argument('--ploss_type', type=str, default='map', help="Ploss Type, \"mseloss | logistic | map\"")

    # GAN Models
    # It first starts with gan weight = 0.1 and when the training epoch reaches 20, gan weight becomes 0.5 and so on.
    parser.add_argument('--noise_dim', type=int, default=16, help="")
    parser.add_argument('--gan_weight', type=float, default=[0.5, 0.7, 1, 1.5, 2.0, 2.5], help="Adversarial Training Alpha")
    parser.add_argument('--gan_weight_schedule', type=float, default=[20, 30, 40, 50, 65, 200], help="Decaying Gan Weight by Epoch")
    parser.add_argument('--disc_hidden', type=int, default=512, help="")  
    parser.add_argument('--disc_dropout', type=float, default=0.5, help="") 

    # Optimization Parameters
    parser.add_argument('--optimizer', type=str, default='adam', help="Optimizer")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for training the model")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--load_ckpt', default=None, help='Load Model Checkpoint')
    parser.add_argument('--start_epoch', type=int, default=1, help='Resume Model Training')

    parser.add_argument('--no_cuda', action="store_true", help="Activate if dont want cuda")
    
    # Model Testing Parameters
    parser.add_argument('--test_partition', type=str,
                        default='test_obs',
                        help="Data partition to perform test")
    parser.add_argument('--test_cache', type=str, help="")
    parser.add_argument('--test_dir', type=str, help="Test output dir")
    parser.add_argument('--test_ckpt', default=None, help="Model Checkpoint for test")
    parser.add_argument('--test_times', type=int, default=10, help='Number of test trials to calculate std.')
    parser.add_argument('--test_render', type=int, default=1, help='Whether to render the outputs as figure')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    configs = EasyDict(vars(parser.parse_args()))
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda')
    configs.ngpus_per_node = torch.cuda.device_count()
    configs.pin_memory = True
    # import pdb; pdb.set_trace()
    config = importlib.import_module(configs.config)
    configs = config.update_config(configs)
    # configs = eval(configs.config).update_configs(configs)
    
    configs.checkpoints_dir = os.path.join(configs.root_dir, 'checkpoints', configs.TAG)
    configs.logs_dir = os.path.join(configs.root_dir, 'logs', configs.TAG)

    if not os.path.isdir(configs.checkpoints_dir):
        os.makedirs(configs.checkpoints_dir)
    if not os.path.isdir(configs.logs_dir):
        os.makedirs(configs.logs_dir)

    return configs

if __name__=="__main__":
    parse_train_configs()