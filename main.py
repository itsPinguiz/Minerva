import argparse
import logging
import os
import subprocess
import sys

from omegaconf import OmegaConf

# ── logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("WardrobeMaster")


def main():
    parser = argparse.ArgumentParser(description="Wardrobe ML Backend - Master Entrypoint")
    # Position argument for mode (train or predict)
    parser.add_argument(
        "mode", type=str, choices=["train", "predict"], help="Execution mode: 'train' or 'predict'"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")

    # Arguments for inference
    parser.add_argument("--image", type=str, help="Path of the image to analyze (required for predict)")

    # Arguments for training
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--bs", type=int, help="Override batch size")

    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    config = OmegaConf.load(args.config)
    logger.info("Loaded config from %s", args.config)

    if args.mode == "train":
        logger.info("🚀 Starting TRAINING mode...")
        logger.info(
            "Backbone: %s | Device: %s",
            config.model.backbone,
            config.system.device.upper(),
        )

        # Build command for the training script
        # Note: training is now in src/training/train.py
        cmd = ["uv", "run", "python", "src/training/train.py", "--config", args.config]
        if args.resume:
            cmd.append("--resume")
        if args.bs:
            cmd.extend(["--bs", str(args.bs)])

        logger.info("Executing: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error("Training failed with exit code %d", e.returncode)
            sys.exit(e.returncode)

    elif args.mode == "predict":
        if not args.image:
            logger.error("Error: You must specify an image with --image (e.g. --image test.jpg)")
            sys.exit(1)

        logger.info("Starting INFERENCE mode on image: %s ...", args.image)

        # Build command for the inference script
        cmd = [
            "uv",
            "run",
            "python",
            "src/models/inference.py",
            "--image",
            args.image,
            "--config",
            args.config,
        ]

        logger.info("Executing: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error("Inference failed with exit code %d", e.returncode)
            sys.exit(e.returncode)


if __name__ == "__main__":
    main()