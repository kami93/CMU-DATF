import os

TRAIN = True
def update_config(configs):
    if TRAIN:
        return update_config_train(configs)
    else:
        return update_config_test(configs)

def update_config_test(configs):
        # Dataset
    configs.intrinsic_rate = 10
    configs.max_distance = 56
    configs.multi_agent = True
    configs.use_scene = True
    configs.scene_size = (64, 64)
    configs.sample_stride = 1 
    configs.sampling_rate = 2
    configs.shuffle = False 
    configs.pin_memory = False 
    configs.device = "cuda"

    # Model 
    configs.model_name = "R2P2_RNN"
    configs.TAG = "r2p2_rnn"
    configs.map_version = '2.1'
    configs.scene_channels = 5 if configs.map_version == '2.1' else 3
    configs.velocity_const = 0.5 
    configs.num_candidates = 12
    configs.decoding_steps = 6
    configs.nfuture = 12
    
    # Paths
    configs.root_dir = "/data/datasets/datf/CMU-DATF"
    configs.cache_path = "/data/datasets/datf/CMU-DATF/caches"
    configs.exp_path = "/data/datasets/datf/CMU-DATF/exps"
    configs.test_ckpt = "../checkpoints/carla_MATF_D128.pth.tar"

    # Test
    configs.batch_size = 1
    configs.test_set = False 
    return configs

def update_config_train(configs):

    # Dataset
    configs.dataset="argoverse_r2p2"
    configs.device = "cuda"
    configs.ploss_criterion = "MSE" 
    configs.intrinsic_rate = 10
    configs.max_distance = 56
    configs.multi_agent = True
    configs.use_scene = True
    configs.scene_size = (64, 64)
    configs.sample_stride = 1 
    configs.sampling_rate = 2
    configs.shuffle = False 
    configs.pin_memory = False 

    # Model
    # configs.model_name = "R2P2_RNN"
    configs.model_name = "R2P2_SimpleRNN"
    configs.TAG = "r2p2_rnn_simple"
    configs.map_version = '2.0'
    configs.scene_channels = 5 if configs.map_version == '2.1' else 1
    configs.velocity_const = 0.5 
    configs.num_candidates = 12
    configs.decoding_steps = 6
    configs.ploss_type = 'mseloss' # 'interpolated_ploss'
    configs.nfuture = 12
    configs.generative=False
    # configs.beta = 0.0
    configs.beta = 0.1

    # Paths
    configs.root_dir = "/data/datasets/datf/CMU-DATF"
    configs.cache_path = "/data/datasets/datf/CMU-DATF/caches"
    configs.exp_path = "/data/datasets/datf/CMU-DATF/exps"

    # Train
    configs.optimizer_name = "adam"
    configs.batch_size = 4
    configs.train = True
    configs.validate = True
    configs.num_epochs = 1
    return configs

if __name__=="__main__":
    print("Call update_config method")
