import os
import random
import utils.file_utils as file_utils
import utils.image_utils as image_utils
import subprocess
import shutil
from typing import List
from cog import BasePredictor, Input, Path

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import threestudio
from threestudio.utils.misc import get_rank
from threestudio.systems.base import BaseSystem
from threestudio.utils.callbacks import ConfigSnapshotCallback
from threestudio.utils.config import ExperimentConfig, load_config
import trimesh


WEIGHTS_CACHE_DIR = "/src/weights"
os.environ["HF_HOME"] = os.environ["HUGGINGFACE_HUB_CACHE"] = WEIGHTS_CACHE_DIR
N_GPUS = 1


CONFIG_PATHS = {
    "with-shading": "configs/imagedream-sd21-shading.yaml",
    "without-shading": "configs/imagedream-sd21-no-shading.yaml",
}
GUIDANCE_CONFIG_PATH = (
    "imagedream/configs/sd_v2_base_ipmv.yaml"  # with pixel controller
)
CKPT_PATH = "weights/imagedream/sd-v2.1-base-4view-ipmv.pt"
BG_REMOVAL_MODEL_PATH = "weights/background_removal/u2net.onnx"


class Predictor(BasePredictor):
    def setup(self) -> None:
        MODEL_FILES_MAP = {
            "Imagedream-SD2.1": {
                "url": "https://weights.replicate.delivery/default/imagedream/imagedream-sd-v2.1-base-4view.tar",
                "cache_dir": "weights/imagedream",
            },
            "CLIP-ViT-H-14-laion2B": {
                "url": "https://weights.replicate.delivery/default/mvdream/clip_vit_h14_laion2B.tar",
                "cache_dir": "weights/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K",
            },
            "SD-2.1-Base-text_encoder-tokenizer": {
                "url": "https://weights.replicate.delivery/default/mvdream/sd-2-1-text-encoder.tar",
                "cache_dir": "weights/models--stabilityai--stable-diffusion-2-1-base",
            },
        }

        # Download model weights if their cache directory doesn't exist
        for _, v in MODEL_FILES_MAP.items():
            if not os.path.exists(v["cache_dir"]):
                file_utils.download_and_extract(url=v["url"], dest=v["cache_dir"])

    def train_model(
        self,
        config_path,
        image_path,
        prompt,
        negative_prompt,
        guidance_scale,
        max_steps,
        n_gpus,
    ) -> str:
        extras = [
            f"system.prompt_processor.prompt={prompt}",
            f"system.prompt_processor.negative_prompt={negative_prompt}",
            f"system.prompt_processor.image_path={image_path}",
            f"system.guidance.guidance_scale={guidance_scale}",
            f"system.guidance.ckpt_path={CKPT_PATH}",
            f"system.guidance.config_path={GUIDANCE_CONFIG_PATH}",
            f"trainer.max_steps={max_steps}",
        ]

        cfg: ExperimentConfig = load_config(config_path, cli_args=extras, n_gpus=n_gpus)
        dm = threestudio.find(cfg.data_type)(cfg.data)
        system: BaseSystem = threestudio.find(cfg.system_type)(
            cfg.system, resumed=cfg.resume is not None
        )
        system.set_save_dir(os.path.join(cfg.trial_dir, "save"))

        callbacks = [
            ModelCheckpoint(
                dirpath=os.path.join(cfg.trial_dir, "ckpts"), **cfg.checkpoint
            ),
            ConfigSnapshotCallback(
                config_path,
                cfg,
                os.path.join(cfg.trial_dir, "configs"),
                use_version=False,
            ),
        ]
        trainer = Trainer(
            logger=False,
            callbacks=callbacks,
            inference_mode=False,
            accelerator="gpu",
            devices=-1,
            **cfg.trainer,
        )

        trainer.fit(system, datamodule=dm, ckpt_path=cfg.resume)
        trainer.test(system, datamodule=dm)
        return cfg.trial_dir

    @torch.no_grad()
    def export_meshes(self, ckpt_path, parsed_config_path, n_gpus) -> None:
        # Initialize the model
        extras = [
            "system.exporter_type=mesh-exporter",
            f"resume={ckpt_path}",
            "system.exporter.context_type=cuda",
        ]
        cfg: ExperimentConfig = load_config(
            parsed_config_path, cli_args=extras, n_gpus=n_gpus
        )

        dm = threestudio.find(cfg.data_type)(cfg.data)
        system: BaseSystem = threestudio.find(cfg.system_type)(cfg.system, resumed=True)
        system.set_save_dir(os.path.join(cfg.trial_dir, "save"))
        trainer = Trainer(
            inference_mode=True,
            accelerator="gpu",
            devices=-1,
            **cfg.trainer,
        )

        # Load the model weights to gpu, otherwise the model will be loaded to cpu which may result in OOM issues
        ckpt = torch.load(ckpt_path, map_location="cuda:0")
        system.set_resume_status(ckpt["epoch"], ckpt["global_step"])

        # Generate the mesh
        trainer.predict(system, datamodule=dm, ckpt_path=cfg.resume)

        # Zip the mesh files
        no_of_iters = cfg.trainer["max_steps"]
        export_obj_path = os.path.join(
            cfg.trial_dir, "save", f"it{no_of_iters}-export", "model.obj"
        )

        return export_obj_path

    def predict(
        self,
        image: Path = Input(
            description="Image to generate a 3D object from.",
        ),
        prompt: str = Input(
            description="Prompt to generate a 3D object.",
        ),
        negative_prompt: str = Input(
            description="Prompt for the negative class. If not specified, a random prompt will be used.",
            default="ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions",
        ),
        guidance_scale: float = Input(
            description="The scale of the guidance loss. Higher values will result in more accurate meshes but may also result in artifacts.",
            ge=1.0,
            le=50.0,
            default=5.0,
        ),
        shading: bool = Input(
            description="Whether to use shading in the generated 3D object. ~40% slower but higher quality with shading.",
            default=False,
        ),
        num_steps: int = Input(
            description="Number of iterations to run the model for.",
            ge=5000,
            le=15000,
            default=12500,
        ),
        seed: int = Input(
            description="The seed to use for the generation. If not specified, a random value will be used.",
            default=None,
        ),
    ) -> Path:
        # Set seed for all random number generators
        if seed is None:
            random.seed()  # Seed from current time
            seed = random.randint(0, 2**32 - 1)
        pl.seed_everything(seed + get_rank(), workers=True)

        # Set config path based on shading
        config_path = (
            CONFIG_PATHS["with-shading"] if shading else CONFIG_PATHS["without-shading"]
        )

        # 0. Preprocess the image
        image = image_utils.preprocess(
            image_path=str(image),
            model_path=BG_REMOVAL_MODEL_PATH,
            remove_bg=True,
            img_size=256,
            border_ratio=0.2,
            recenter=True,
        )

        # 1. Generate images and train the 2Dto3D model
        trial_dir = self.train_model(
            config_path=config_path,
            image_path=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            max_steps=num_steps,
            guidance_scale=guidance_scale,
            n_gpus=N_GPUS,
        )

        ckpt_path = os.path.join(trial_dir, "ckpts", "last.ckpt")
        parsed_config_path = os.path.join(trial_dir, "configs", "parsed.yaml")

        # 2. Export the mesh
        export_obj_path = self.export_meshes(
            ckpt_path=ckpt_path,
            parsed_config_path=parsed_config_path,
            n_gpus=N_GPUS,
        )

        # 3. Prepare .glb file and return
        mesh = trimesh.load(export_obj_path, process=False)
        out_mesh_path = "mesh.glb"
        mesh.export(out_mesh_path)

        if os.path.exists("lightning_logs"):
            shutil.rmtree("lightning_logs")

        return Path(out_mesh_path)
