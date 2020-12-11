import torch
import torch.nn as nn
import torch.nn.functional as F
from .matf import SimpleEncoderDecoder, SocialPooling, MATF
from .cam import CAM, CAM_NFDecoder, Scene_CAM_NFDecoder, Global_Scene_CAM_NFDecoder

from .matf_gan import MATF_Gen, MATF_Disc
from .r2p2_ma import R2P2_SimpleRNN, R2P2_RNN
from .desire import DESIRE_SGM, DESIRE_IOC
import os

__all__ = {
    "SimpleEncoderDecoder": SimpleEncoderDecoder,
    "SocialPooling": SocialPooling,
    "MATF": MATF,
    "MATF_GAN": [MATF_Gen, MATF_Disc],
    "R2P2_SimpleRNN": R2P2_SimpleRNN,
    "R2P2_RNN": R2P2_RNN,
    "DESIRE": [DESIRE_SGM, DESIRE_IOC],
    "CAM": CAM,
    "CAM_NFDecoder": CAM_NFDecoder,
    "Scene_CAM_NFDecoder": Scene_CAM_NFDecoder,
    "Global_Scene_CAM_NFDecoder": Global_Scene_CAM_NFDecoder,
    "AttGlobal_Scene_CAM_NFDecoder": Global_Scene_CAM_NFDecoder,
    # "Scene_CAM": Scene_CAM, 
    # "Scene_CAM_NFDecoder": Scene_CAM_NFDecoder,

}

class build_model(nn.Module):
    def __init__(self, **kwargs):
        super(build_model, self).__init__()
        self.cfg=kwargs.get('cfg')
        cfg = self.cfg
        model_fn = __all__[self.cfg.model_name]
        if isinstance(model_fn, list):
            self.model = []
            for model_trace in model_fn:
                self.model.append( 
                    model_trace(cfg=cfg, device=cfg.device)
                 )
        else:    
            self.model = model_fn(cfg=cfg, device=cfg.device)

    def forward(self, batch, select=None):
        if isinstance(batch, dict):
            if isinstance(self.model, list) and select:
                pred_out = self.model[select](**batch)
            else:
                pred_out = self.model(**batch)
        else:
            assert "Please construct dict of the data"
        return pred_out

    def load_params_from_file(self, filenames, logger=None, to_cpu=False):

        if not isinstance(filenames, list):
            filenames = [filenames]

        for i, filename in enumerate(filenames): 
            if not os.path.isfile(filename):
                raise FileNotFoundError
            
            if logger is not None:
                logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
            loc_type = torch.device('cpu') if to_cpu else None
            checkpoint = torch.load(filename, map_location=loc_type)
            model_state_disk = checkpoint['model_state']

            if isinstance(self.model, list):
                self.model_cur = self.model[i]
            else:
                self.model_cur = self.model

            update_model_state = {}
            for key, val in model_state_disk.items():
                if key in self.model_cur.state_dict() and self.model_cur.state_dict()[key].shape == model_state_disk[key].shape:
                    update_model_state[key] = val
                    # logger.info('Update weight %s: %s' % (key, str(val.shape)))

            state_dict = self.model_cur.state_dict()
            state_dict.update(update_model_state)
            self.model_cur.load_state_dict(state_dict, strict=False)
            
            if logger is not None:
                for key in state_dict:
                    if key not in update_model_state:
                        logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

                logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.model_cur.state_dict())))

            