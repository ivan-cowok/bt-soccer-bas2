import dataclasses
import random

import timm
import torch
from torch import nn
import torchvision.transforms.v2 as T


from collections import OrderedDict

from dudek.ml.model.tdeed.modules.layers import EDSGPMIXERLayers, FCLayers
from dudek.ml.model.tdeed.modules.shift import make_temporal_shift


@dataclasses.dataclass
class TDeedLoss:
    total_loss: float
    ce_labels_loss: float
    mse_displacement_loss: float
    bce_loss_teams: float


class AddGaussianNoise(nn.Module):
    """Per-frame additive Gaussian noise.

    Sigma is sampled once per call (i.e. once per clip) so that the noise
    *level* is constant across the clip — that matches what a single recording
    looks like (sensor noise level depends on the camera/lighting, not the
    instant). The noise tensor itself is independent for every frame, every
    pixel, every channel — sensor noise / snow grain decorrelates across
    frames.

    Expects input in [0.0, 1.0].
    """

    def __init__(self, sigma_range: tuple[float, float] = (0.0, 0.03)):
        super().__init__()
        self.sigma_min, self.sigma_max = sigma_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        if sigma <= 0.0:
            return x
        noise = torch.randn_like(x) * sigma
        return (x + noise).clamp_(0.0, 1.0)


class TDeedModule(nn.Module):

    def __init__(
        self,
        clip_len: int,
        n_layers: int,
        sgp_ks: int,
        sgp_k: int,
        num_classes: int,
        features_model_name: str = "regnety_002",
        temporal_shift_mode: str = "gsf",
        gaussian_blur_ks: int = 3,
        grad_checkpointing: bool = False,
    ):
        super().__init__()

        self.features_model_name = features_model_name
        self.temporal_shift_mode = temporal_shift_mode
        self.sgp_k = sgp_k
        self.sgp_ks = sgp_ks
        self.n_layers = n_layers
        self.grad_checkpointing = grad_checkpointing

        features = timm.create_model(
            features_model_name,
            pretrained=True,
        )
        if grad_checkpointing and hasattr(features, "set_grad_checkpointing"):
            features.set_grad_checkpointing(True)

        feat_dim = features.get_classifier().in_features
        features.reset_classifier(0)

        self._d = feat_dim

        self._require_clip_len = clip_len
        make_temporal_shift(features, clip_len, mode=temporal_shift_mode)

        self._features = features
        self._feat_dim = self._d
        feat_dim = self._d

        # Positional encoding
        self.temp_enc = nn.Parameter(
            torch.normal(mean=0, std=1 / clip_len, size=(clip_len, self._d))
        )
        self._temp_fine = EDSGPMIXERLayers(
            feat_dim,
            clip_len,
            num_layers=n_layers,
            ks=sgp_ks,
            k=sgp_k,
            concat=True,
        )
        self._pred_fine = FCLayers(self._feat_dim, num_classes + 1)
        self._pred_displ = FCLayers(self._feat_dim, 1)

        # Augmentations.
        #
        # Notes:
        #  * GaussianBlur sigma capped at 1.0 — at our 640x360 resolution a
        #    6-10 px ball survives sigma <= 1.0 but loses contrast badly above.
        #    Default torchvision sigma range is (0.1, 2.0) which is destructive
        #    for small balls; (0.3, 1.0) is the safer band for our domain.
        #  * RandomErasing simulates real-world occlusions (player walking past
        #    the ball, defender between camera and ball, sideline observer).
        #    Same box across all frames in a clip via v2's per-call param
        #    sampling. scale=(0.005, 0.03) keeps boxes small (0.5-3% of frame
        #    area, ≈30x40 to 110x100 px at 640x360) — large enough to be
        #    realistic, small enough that erasing the ball is a low-probability
        #    event (~0.2% of training frames). Forces the model to use temporal
        #    context and player-action cues instead of ball-only shortcuts.
        #  * AddGaussianNoise simulates sensor / snow / low-quality grain that
        #    appears in our footage. Independent noise per frame, sigma fixed
        #    per clip. Applied last so it is independent of color/blur.
        # ColorJitter ranges tuned for our domain (low-league, fixed wide camera,
        # high real-world photometric variability — snow, daylight, evening,
        # children's matches, mismatched broadcast quality):
        #   - hue=0.05 (tightened from 0.1): real white-balance variation is
        #     small; ±10% hue rotation makes grass purple-ish, which is
        #     unphysical and wastes capacity.
        #   - saturation=(0.5, 1.3) (widened): snow matches need the low end,
        #     saturated daylight needs the high end.
        #   - brightness=(0.6, 1.3) (widened): snow is very bright, evening is
        #     dark; need both tails covered.
        #   - contrast=(0.8, 1.15) (tightened): low contrast on already-low-
        #     quality footage erases ball-vs-grass detail. Safer to narrow.
        self.augmentation = T.Compose(
            [
                T.RandomApply([T.ColorJitter(hue=0.05)], p=0.25),
                T.RandomApply([T.ColorJitter(saturation=(0.5, 1.3))], p=0.25),
                T.RandomApply([T.ColorJitter(brightness=(0.6, 1.3))], p=0.25),
                T.RandomApply([T.ColorJitter(contrast=(0.8, 1.15))], p=0.25),
                T.RandomApply(
                    [T.GaussianBlur(kernel_size=gaussian_blur_ks, sigma=(0.3, 1.0))],
                    p=0.20,
                ),
                T.RandomErasing(
                    p=0.15, scale=(0.005, 0.03), ratio=(0.3, 3.3), value=0
                ),
                T.RandomApply([AddGaussianNoise(sigma_range=(0.0, 0.03))], p=0.20),
            ]
        )

        # Standarization
        self.standarization = T.Compose(
            [
                T.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                )  # Imagenet mean and std
            ]
        )

    def forward(self, x, y=None, inference=False):

        x = self.normalize(x)  # Normalize to 0-1
        batch_size, clip_len, channels, height, width = x.shape

        if not inference:
            x.view(-1, channels, height, width)
            x = x.view(batch_size, clip_len, channels, height, width)
            x = self.augment(x)  # augmentation per-batch
            x = self.standarize(x)  # standarization imagenet stats

        else:
            x = x.view(-1, channels, height, width)
            x = x.view(batch_size, clip_len, channels, height, width)
            x = self.standarize(x)

        # Extract features
        im_feat = self._features(x.view(-1, channels, height, width)).reshape(
            batch_size, clip_len, self._d
        )

        # Temporal encoding
        im_feat = im_feat + self.temp_enc.expand(batch_size, -1, -1)

        # Temporal module (SGP-Mixer)

        output_data = {}
        im_feat = self._temp_fine(im_feat)

        displ_feat = self._pred_displ(im_feat).squeeze(-1)
        output_data["displ_feat"] = displ_feat

        im_feat = self._pred_fine(im_feat)

        output_data["im_feat"] = im_feat

        return output_data, y

    def normalize(self, x):
        return x / 255.0

    def augment(self, x):
        for i in range(x.shape[0]):
            x[i] = self.augmentation(x[i])
        return x

    def standarize(self, x):
        for i in range(x.shape[0]):
            x[i] = self.standarization(x[i])
        return x

    def load_backbone(self, model_weight_path: str):
        m = torch.load(
            model_weight_path, map_location=torch.device("cpu"), weights_only=True
        )
        _features_layers = OrderedDict(
            {
                k[len("_features.") :]: v
                for k, v in m.items()
                if k.startswith("_features.")
            }
        )
        self._features.load_state_dict(_features_layers)
        _temp_fine_layers = OrderedDict(
            {
                k[len("_temp_fine.") :]: v
                for k, v in m.items()
                if k.startswith("_temp_fine.")
            }
        )
        self._temp_fine.load_state_dict(_temp_fine_layers)

    def freeze_backbone(self):
        for param in self._features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self._features.parameters():
            param.requires_grad = True

    def load_all(self, model_weight_path: str):
        m = torch.load(
            model_weight_path, map_location=torch.device("cpu"), weights_only=True
        )
        self.load_state_dict(m)
