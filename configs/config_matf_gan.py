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
    configs.scene_size = (60, 60)
    configs.sample_stride = 1 
    configs.sampling_rate = 2
    configs.shuffle = False 
    configs.pin_memory = False 

    # Model 
    configs.model_name = "MATF"
    configs.TAG = "matf"
    configs.device = "cuda"
    configs.agent_embed_dim = 128
    configs.nfuture = 12 # redundant if using sampling ratesa
    configs.lstm_layers = 1
    configs.lstm_dropout = 0.3
    configs.encoder_type="ShallowCNN" # "ShallowCNN | ResNet"
    configs.scene_channels= 3
    configs.scene_dropout=0.5
    configs.freeze_resnet=True
    configs.disc_hidden = 512
    configs.disc_dropout = 0.5
    configs.gan_weight_schedule = [20, 30, 40, 50, 65, 200]
    configs.gan_weight = [0.5, 0.7, 1, 1.5, 2.0, 2.5]
    
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
    configs.device = "cuda"
     
    configs.intrinsic_rate = 10
    configs.max_distance = 56
    configs.multi_agent = True
    configs.use_scene = True
    configs.scene_size = (60, 60)
    configs.sample_stride = 1 
    configs.sampling_rate = 2
    configs.shuffle = False 
    configs.pin_memory = False 

    # Model
    configs.model_name = "MATF_GAN"
    configs.TAG = "matf_gan"
    configs.agent_embed_dim = 128
    configs.nfuture = 12 # redundant if using sampling ratesa
    configs.lstm_layers = 1
    configs.lstm_dropout = 0.3
    configs.encoder_type="ShallowCNN" # "ShallowCNN | ResNet"
    configs.freeze_resnet=True
    configs.scene_channels= 1
    configs.scene_dropout=0.5
    configs.disc_hidden = 512
    configs.disc_dropout = 0.5
    configs.gan_weight_schedule = [20, 30, 40, 50, 65, 200]
    configs.gan_weight = [0.5, 0.7, 1, 1.5, 2.0, 2.5]
    configs.generative = True
    configs.map_version = '2.0'
    configs.scene_channels = 5 if configs.map_version == '2.1' else 3

    # Paths
    configs.root_dir = "/data/datasets/datf/CMU-DATF"
    configs.cache_path = "/data/datasets/datf/CMU-DATF/caches"
    configs.exp_path = "/data/datasets/datf/CMU-DATF/exps"

    # Train
    configs.optimizer_name = ["adam", "adam"] # Generator and Discriminator
    configs.batch_size = 2
    configs.train = True
    configs.validate = True
    return configs

if __name__=="__main__":
    print("Call update_config method")
