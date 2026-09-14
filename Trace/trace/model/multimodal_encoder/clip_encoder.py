import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):

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

        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def _native_image_hw(self):
        image_size = self.config.image_size
        if isinstance(image_size, (tuple, list)):
            return int(image_size[0]), int(image_size[1])
        return int(image_size), int(image_size)

    def _encode_selected_features(self, pixel_values):
        """Encode native images or a full-resolution vertical comparison.

        A paired teacher frame is represented as one ``[C, 2H, W]`` tensor so
        the input contract remains explicitly upper-reference/lower-candidate.
        Running CLIP attention over the doubled-height patch grid needlessly
        squares the attention memory and exceeds 24 GB cards.  Encode the two
        native 24x24 grids separately, then concatenate their selected patch
        features in upper/lower order.  This keeps every input pixel and the
        same ordered patch tokens while bounding peak CLIP attention at
        its pretrained native resolution.
        """
        native_h, native_w = self._native_image_hw()
        if tuple(pixel_values.shape[-2:]) == (2 * native_h, native_w):
            upper_pixels = pixel_values[..., :native_h, :]
            lower_pixels = pixel_values[..., native_h:, :]

            upper_outputs = self._forward_vision(upper_pixels)
            upper_features = self.feature_select(upper_outputs)
            del upper_outputs

            lower_outputs = self._forward_vision(lower_pixels)
            lower_features = self.feature_select(lower_outputs)
            del lower_outputs
            return torch.cat([upper_features, lower_features], dim=1)

        return self.feature_select(self._forward_vision(pixel_values))

    def _forward_with_interpolated_positions(self, pixel_values):
        """Run CLIP on a non-native grid without resampling the input pixels.

        TRACE pins transformers 4.40.1, whose ``CLIPVisionModel.forward`` does
        not expose the later ``interpolate_pos_encoding`` argument.  This is
        the same two-dimensional bicubic positional-embedding interpolation
        used by modern ViT/CLIP implementations, kept locally so the existing
        TRACE checkpoint and dependency versions remain compatible.  It is
        retained as a compatibility path for other non-native image shapes.
        Exact upper-reference/lower-candidate canvases are split into two
        native CLIP calls by ``_encode_selected_features`` instead.
        """
        vision_model = self.vision_tower.vision_model
        embeddings = vision_model.embeddings
        patch_embeddings = embeddings.patch_embedding(pixel_values)
        batch_size, _, grid_h, grid_w = patch_embeddings.shape
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError(f"invalid CLIP patch grid {(grid_h, grid_w)}")

        # CLIP's learned table was trained on a square source grid.  Infer the
        # grid from the table rather than hard-coding 24x24 / 336px.
        position_weight = embeddings.position_embedding.weight
        source_patch_count = position_weight.shape[0] - 1
        source_side = int(source_patch_count ** 0.5)
        if source_side * source_side != source_patch_count:
            raise RuntimeError(
                "CLIP positional table is not square; cannot safely construct "
                f"a 2-D interpolated grid from {source_patch_count} patch positions"
            )

        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)
        class_embedding = embeddings.class_embedding.expand(batch_size, 1, -1)
        source_positions = position_weight[1:].reshape(1, source_side, source_side, -1).permute(0, 3, 1, 2)
        patch_positions = F.interpolate(
            source_positions.float(), size=(grid_h, grid_w), mode="bicubic", align_corners=False
        ).to(dtype=patch_embeddings.dtype)
        patch_positions = patch_positions.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)
        positions = torch.cat([position_weight[:1].to(dtype=patch_embeddings.dtype).unsqueeze(0), patch_positions], dim=1)

        hidden_states = torch.cat([class_embedding, patch_embeddings], dim=1) + positions
        hidden_states = vision_model.pre_layrnorm(hidden_states)
        return vision_model.encoder(
            inputs_embeds=hidden_states,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

    def _forward_vision(self, pixel_values):
        """Keep TRACE's native path exact and interpolate only when necessary."""
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        native_h, native_w = self._native_image_hw()
        if tuple(pixel_values.shape[-2:]) == (native_h, native_w):
            return self.vision_tower(pixel_values, output_hidden_states=True)
        patch_size = self.config.patch_size
        if pixel_values.shape[-2] % patch_size or pixel_values.shape[-1] % patch_size:
            raise ValueError(
                f"non-native CLIP input {tuple(pixel_values.shape[-2:])} must be divisible by patch size {patch_size}"
            )
        return self._forward_with_interpolated_positions(pixel_values)

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self._encode_selected_features(image.unsqueeze(0)).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_features = self._encode_selected_features(images).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size
