import os
import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers import AutoImageProcessor, AutoModel, AutoConfig

class DINOVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()


        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.clip_vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)

        dino_path = os.environ["DINO_PATH"]
        self.vision_tower = torch.hub.load(dino_path, 'dinov2_vitl14', source='local')

        self.clip_vision_tower.requires_grad_(False)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs["x_prenorm"]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        image_forward_outs, attn_map = self.vision_tower.forward_features(images.to(device=self.device, dtype=self.dtype), return_attention=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features,attn_map

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.clip_vision_tower.dtype

    @property
    def device(self):
        return self.clip_vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.clip_vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        #return self.config.hidden_size
        return 1024

    @property
    def num_patches(self):
        #return (self.config.image_size // self.config.patch_size) ** 2
        return 256

