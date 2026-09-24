from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel
from typing_extensions import override

from finetune.schemas import Components
from finetune.trainer import Trainer
from finetune.utils import unwrap_model

from ..utils import register
from ..src.motion_stage1 import Stage1MotionPipeline, build_motion_config_from_args, is_motion_enabled
from functools import partial
from torchvision import transforms
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput

class CAMOS1Trainer(Trainer):
    UNLOAD_LIST = ["text_encoder", "vae"]
    MOTION_ATTR = "camo_motion"

    @override
    def load_components(self) -> Components:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXPipeline

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder"
        )

        components.transformer = CogVideoXTransformer3DModel.from_pretrained(
            model_path, subfolder="transformer"
        )

        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")

        components.scheduler = CogVideoXDPMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )

        return components

    @override
    def initialize_pipeline(self) -> CogVideoXPipeline:
        pipe = CogVideoXPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
        )
        return pipe

    @override
    def prepare_models(self) -> None:
        super().prepare_models()

        self.motion_config = build_motion_config_from_args(self.args)
        self.motion_enabled = is_motion_enabled(self.motion_config)

        if self.motion_enabled:
            latent_channels = int(self.state.transformer_config.in_channels)
            motion_pipeline = Stage1MotionPipeline(
                config=self.motion_config,
                latent_channels=latent_channels,
            )
            setattr(self.components.transformer, self.MOTION_ATTR, motion_pipeline)
            motion_pipeline.attach_dit_reliability_injector(self.components.transformer)

    @override
    def prepare_trainable_parameters(self):
        super().prepare_trainable_parameters()

        motion_pipeline = self._get_motion_pipeline()
        if motion_pipeline is not None:
            motion_pipeline.requires_grad_(True)

    def _get_motion_pipeline(self) -> Optional[Stage1MotionPipeline]:
        transformer = unwrap_model(self.accelerator, self.components.transformer)
        motion_pipeline = getattr(transformer, self.MOTION_ATTR, None)
        if motion_pipeline is None:
            return None
        return motion_pipeline

    @override
    def clear_runtime_contexts(self) -> None:
        motion_pipeline = self._get_motion_pipeline()
        if motion_pipeline is not None:
            motion_pipeline.clear_dit_reliability_context()

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.components.text_encoder(
            prompt_token_ids.to(self.accelerator.device)
        )[0]
        return prompt_embedding

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {"encoded_hq_videos": [], "encoded_lq_videos": [], "hq_videos": [], "lq_videos": [], "prompts": [], "prompt_embeddings": [], "video_metadatas": [], "encoded_video_metadatas": []}

        for sample in samples:
            ret["hq_videos"].append(sample["hq_video"])
            ret["lq_videos"].append(sample["lq_video"])
            ret["prompts"].append(sample["prompt"])
            ret["prompt_embeddings"].append(sample["prompt_embedding"])
            ret["video_metadatas"].append(sample["video_metadata"])
            if sample["encoded_hq_video"] != None:
                ret["encoded_hq_videos"].append(sample["encoded_hq_video"])
            if sample["encoded_lq_video"] != None:
                ret["encoded_lq_videos"].append(sample["encoded_lq_video"])
            if sample["encoded_video_metadata"] != None:
                ret["encoded_video_metadatas"].append(sample["encoded_video_metadata"])

        ret["hq_videos"] = torch.stack(ret["hq_videos"])
        ret["lq_videos"] = torch.stack(ret["lq_videos"])
        ret["prompt_embeddings"] = torch.stack(ret["prompt_embeddings"])
        if len(ret["encoded_hq_videos"]) > 0:
            ret["encoded_hq_videos"] = torch.stack(ret["encoded_hq_videos"])
        if len(ret["encoded_lq_videos"]) > 0:
            ret["encoded_lq_videos"] = torch.stack(ret["encoded_lq_videos"])

        return ret

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        prompt_embedding = batch["prompt_embeddings"]
        lq_videos = batch.get("lq_videos")
        if self.args.is_latent:
            lq_latent = batch["encoded_lq_videos"]
            hq_latent = batch["encoded_hq_videos"]
        else:
            with torch.no_grad():
                self.components.vae.to(self.accelerator.device)
                hq_videos = batch["hq_videos"]
                mix_videos = torch.cat([lq_videos, hq_videos], dim=0).to(self.accelerator.device)
                mix_latent = self.encode_video(mix_videos)
                lq_latent, hq_latent = mix_latent.chunk(2, dim=0)

        motion_pipeline = self._get_motion_pipeline()
        m_obj_lr = None
        m_turb_lr = None
        rho_t = None
        q_stocp_l = None
        motion_loss = None
        motion_loss_dict = {}

        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, C, F, H, W]

        if (
            motion_pipeline is not None
            and motion_pipeline.enable_stocp_uncertainty
            and lq_videos is not None
        ):
            q_stocp_l = motion_pipeline.estimate_stocp_uncertainty(
                lq_video=lq_videos,
                latent_shape_bcfhw=lq_latent.shape,
            )

        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            ncopy = lq_latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            lq_first_frame = lq_latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            lq_latent = torch.cat([lq_first_frame.repeat(1, 1, ncopy, 1, 1), lq_latent], dim=2)
            if q_stocp_l is not None and ncopy > 0:
                q_stocp_l = torch.cat([q_stocp_l[:, :, :1].repeat(1, 1, ncopy, 1, 1), q_stocp_l], dim=2)

            hq_first_frame = hq_latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            hq_latent = torch.cat([hq_first_frame.repeat(1, 1, ncopy, 1, 1), hq_latent], dim=2)

            assert lq_latent.shape[2] % patch_size_t == 0 and hq_latent.shape[2] % patch_size_t == 0

        if motion_pipeline is not None and motion_pipeline.enable_decoupling:
            m_obj_lr, m_turb_lr, motion_loss, motion_loss_dict = motion_pipeline.extract_motion_and_loss(
                hr_latent=hq_latent,
                lq_latent=lq_latent,
                q_stocp_l=q_stocp_l,
            )

        if (
            motion_pipeline is not None
            and (
                motion_pipeline.enable_reliability_injection
                or motion_pipeline.enable_reliability_velocity_scaling
            )
            and m_obj_lr is not None
            and m_turb_lr is not None
        ):
            rho_t = motion_pipeline.estimate_reliability(m_obj_lr, m_turb_lr, q_stocp_l=q_stocp_l)

        if motion_pipeline is not None and motion_pipeline.enable_turbulence_rectifier and m_turb_lr is not None:
            lq_latent = motion_pipeline.rectify_latent(lq_latent, m_turb_lr)

        batch_size, num_channels, num_frames, height, width = lq_latent.shape

        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=lq_latent.dtype)

        lq_latent = lq_latent.permute(0, 2, 1, 3, 4)  # from [B, C, F, H, W] to [B, F, C, H, W]
        reshape_hq_latent = hq_latent.permute(0, 2, 1, 3, 4)  # from [B, C, F, H, W] to [B, F, C, H, W]

        # Noise Step (Select)
        if self.args.noise_step != 0:
            add_timesteps = torch.full(
                (batch_size,),
                fill_value=self.args.noise_step,
                dtype=torch.long,
                device=self.accelerator.device,
            )

            # Add noise to latent
            noise = torch.randn_like(lq_latent)
            lq_latent = self.components.scheduler.add_noise(lq_latent, noise, add_timesteps)
        
        timesteps = torch.full(
            (batch_size,),
            fill_value=self.args.sr_noise_step,
            dtype=torch.long,
            device=self.accelerator.device,
        )

        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        # Predict noise (actual is velocity)
        if (
            motion_pipeline is not None
            and motion_pipeline.enable_reliability_injection
            and rho_t is not None
        ):
            motion_pipeline.prepare_dit_reliability_context(rho_t, lq_latent.shape)

        predicted_noise = self.components.transformer(
            hidden_states=lq_latent,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]
        if (
            motion_pipeline is not None
            and motion_pipeline.enable_reliability_velocity_scaling
            and rho_t is not None
        ):
            predicted_noise = motion_pipeline.scale_velocity_by_reliability(predicted_noise, rho_t)
        
        # Denoise: x0 (x0' = αt x_t - σt * vθ(z_t))
        latent_pred = self.components.scheduler.get_velocity(
            predicted_noise, lq_latent, timesteps
        )
        # get_velocity (function): velocity = sqrt_alpha_prod * input2 - sqrt_one_minus_alpha_prod * input1
        # get_velocity (here): latent_pred = sqrt_alpha_prod * lq_latent - sqrt_one_minus_alpha_prod * predicted_noise

        if motion_pipeline is not None and motion_pipeline.enable_object_motion_enhancement and m_obj_lr is not None:
            latent_pred_bcfhw = latent_pred.permute(0, 2, 1, 3, 4).contiguous()
            latent_pred_bcfhw = motion_pipeline.enhance_latent(latent_pred_bcfhw, m_obj_lr)
            latent_pred = latent_pred_bcfhw.permute(0, 2, 1, 3, 4).contiguous()

        # Calculate loss (MSE)
        recon_loss = F.mse_loss(latent_pred.float(), reshape_hq_latent.float(), reduction="mean")

        total_loss = recon_loss
        if motion_loss is not None:
            total_loss = total_loss + motion_pipeline.motion_loss_weight * motion_loss

        loss_dict = {
            "loss": total_loss.detach().item(),
            "loss_recon": recon_loss.detach().item(),
        }
        if motion_loss is not None:
            loss_dict["loss_motion"] = motion_loss.detach().item()
            loss_dict["loss_motion_weighted"] = (
                motion_pipeline.motion_loss_weight * motion_loss
            ).detach().item()
            for key, value in motion_loss_dict.items():
                if torch.is_tensor(value):
                    value = value.detach().item()
                loss_dict[f"motion_{key}"] = float(value)

        return total_loss, loss_dict

    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: CogVideoXPipeline
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
        """
        prompt, video, ref_video = eval_data["prompt"], eval_data["video_tensor"], eval_data["ref_video"]

        # [F, C, H, W]
        H_, W_ = video.shape[2], video.shape[3]
        video = torch.nn.functional.interpolate(video, size=(H_*4, W_*4), mode="bilinear", align_corners=False)
        self.__frame_transform = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)] # -1, 1
        )
        video = torch.stack([self.__frame_transform(f) for f in video], dim=0)
        video = video.unsqueeze(0)
        # [B, C, F, H, W]
        video = video.permute(0, 2, 1, 3, 4).contiguous()

        motion_pipeline = self._get_motion_pipeline()

        with torch.no_grad():
            self.components.vae.to(self.accelerator.device)
            latent = self.encode_video(video)
            m_obj_lr = None
            m_turb_lr = None
            rho_t = None
            q_stocp_l = None

            if motion_pipeline is not None and motion_pipeline.enable_stocp_uncertainty:
                q_stocp_l = motion_pipeline.estimate_stocp_uncertainty(
                    lq_video=video,
                    latent_shape_bcfhw=latent.shape,
                )

            patch_size_t = self.state.transformer_config.patch_size_t
            if patch_size_t is not None:
                ncopy = latent.shape[2] % patch_size_t
                # Copy the first frame ncopy times to match patch_size_t
                first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
                latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
                if q_stocp_l is not None and ncopy > 0:
                    q_stocp_l = torch.cat([q_stocp_l[:, :, :1].repeat(1, 1, ncopy, 1, 1), q_stocp_l], dim=2)

                assert latent.shape[2] % patch_size_t == 0

            if motion_pipeline is not None and motion_pipeline.enable_decoupling:
                m_obj_lr, m_turb_lr = motion_pipeline.extract_lr_motion(latent)

            if (
                motion_pipeline is not None
                and (
                    motion_pipeline.enable_reliability_injection
                    or motion_pipeline.enable_reliability_velocity_scaling
                )
                and m_obj_lr is not None
                and m_turb_lr is not None
            ):
                rho_t = motion_pipeline.estimate_reliability(m_obj_lr, m_turb_lr, q_stocp_l=q_stocp_l)

            if (
                motion_pipeline is not None
                and motion_pipeline.enable_turbulence_rectifier
                and m_turb_lr is not None
            ):
                latent = motion_pipeline.rectify_latent(latent, m_turb_lr)

            batch_size, num_channels, num_frames, height, width = latent.shape

            # Get prompt embeddings
            self.components.text_encoder.to(self.accelerator.device)
            prompt_embedding = self.encode_text(prompt)
            _, seq_len, _ = prompt_embedding.shape
            prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

            latent = latent.permute(0, 2, 1, 3, 4)

            # Add noise to latent (Select)
            if self.args.noise_step != 0:
                noise = torch.randn_like(latent)
                add_timesteps = torch.full(
                    (batch_size,),
                    fill_value=self.args.noise_step,
                    dtype=torch.long,
                    device=self.accelerator.device,
                )
                latent = self.components.scheduler.add_noise(latent, noise, add_timesteps)

            timesteps = torch.full(
                (batch_size,),
                fill_value=self.args.sr_noise_step,
                dtype=torch.long,
                device=self.accelerator.device,
            )

            # Prepare rotary embeds
            vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
            transformer_config = self.state.transformer_config
            rotary_emb = (
                self.prepare_rotary_positional_embeddings(
                    height=height * vae_scale_factor_spatial,
                    width=width * vae_scale_factor_spatial,
                    num_frames=num_frames,
                    transformer_config=transformer_config,
                    vae_scale_factor_spatial=vae_scale_factor_spatial,
                    device=self.accelerator.device,
                )
                if transformer_config.use_rotary_positional_embeddings
                else None
            )

            # Predict noise (actual is velocity)
            if (
                motion_pipeline is not None
                and motion_pipeline.enable_reliability_injection
                and rho_t is not None
            ):
                motion_pipeline.prepare_dit_reliability_context(rho_t, latent.shape)
            try:
                predicted_noise = self.components.transformer(
                    hidden_states=latent,
                    encoder_hidden_states=prompt_embedding,
                    timestep=timesteps,
                    image_rotary_emb=rotary_emb,
                    return_dict=False,
                )[0]
            finally:
                if motion_pipeline is not None:
                    motion_pipeline.clear_dit_reliability_context()
            if (
                motion_pipeline is not None
                and motion_pipeline.enable_reliability_velocity_scaling
                and rho_t is not None
            ):
                predicted_noise = motion_pipeline.scale_velocity_by_reliability(predicted_noise, rho_t)
            
            # Denoise: x0 (x0' = αt x_t - σt * vθ(z_t))
            latent_generate = self.components.scheduler.get_velocity(
                predicted_noise, latent, timesteps
            )
            # get_velocity (function): velocity = sqrt_alpha_prod * input2 - sqrt_one_minus_alpha_prod * input1
            # get_velocity (here): latent_pred = sqrt_alpha_prod * latent - sqrt_one_minus_alpha_prod * predicted_noise

            if (
                motion_pipeline is not None
                and motion_pipeline.enable_object_motion_enhancement
                and m_obj_lr is not None
            ):
                latent_generate_bcfhw = latent_generate.permute(0, 2, 1, 3, 4).contiguous()
                latent_generate_bcfhw = motion_pipeline.enhance_latent(latent_generate_bcfhw, m_obj_lr)
                latent_generate = latent_generate_bcfhw.permute(0, 2, 1, 3, 4).contiguous()

            # generate video
            if patch_size_t is not None and ncopy > 0:
                latent_generate = latent_generate[:, ncopy:, :, :, :]
            video = pipe.decode_latents(latent_generate)

            # # output video as PIL list
            # video = pipe.video_processor.postprocess_video(video=video, output_type='pil')
            # video = CogVideoXPipelineOutput(frames=video).frames[0]

            # output video as tensor [B, C, F, H, W]
            video = (video * 0.5 + 0.5).clamp(0.0, 1.0)
            
        return [("video", video[0])]
    
    # Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
    def get_resize_crop_region_for_grid(self, src, tgt_width, tgt_height):
        tw = tgt_width
        th = tgt_height
        h, w = src
        r = h / w
        if r > (th / tw):
            resize_height = th
            resize_width = int(round(th / h * w))
        else:
            resize_width = tw
            resize_height = int(round(tw / w * h))

        crop_top = int(round((th - resize_height) / 2.0))
        crop_left = int(round((tw - resize_width) / 2.0))

        return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: Dict,
        vae_scale_factor_spatial: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        p = transformer_config.patch_size
        p_t = transformer_config.patch_size_t

        base_size_width = transformer_config.sample_width // p
        base_size_height = transformer_config.sample_height // p

        if p_t is None:
            # CogVideoX 1.0
            grid_crops_coords = self.get_resize_crop_region_for_grid(
                (grid_height, grid_width), base_size_width, base_size_height
            )
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=transformer_config.attention_head_dim,
                crops_coords=grid_crops_coords,
                grid_size=(grid_height, grid_width),
                temporal_size=num_frames,
                device=device,
            )
        else:
            # CogVideoX 1.5
            base_num_frames = (num_frames + p_t - 1) // p_t

            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=transformer_config.attention_head_dim,
                crops_coords=None,
                grid_size=(grid_height, grid_width),
                temporal_size=base_num_frames,
                grid_type="slice",
                max_size=(base_size_height, base_size_width),
                device=device,
            )

        return freqs_cos, freqs_sin

register("camo", "lora", CAMOS1Trainer)
