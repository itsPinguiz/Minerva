import argparse
from omegaconf import OmegaConf

def main():
    parser = argparse.ArgumentParser(description="Wardrobe ML Backend - Training & Inference Entrypoint")
    parser.add_argument("--mode", type=str, choices=["train", "predict"], default="train", help="Execution mode")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    config = OmegaConf.load(args.config)
    print(f"Loaded config from {args.config}")
    
    if args.mode == "train":
        print(f"Starting training with backbone: {config.model.backbone} on device: {config.system.device}")
        print(f"Epochs: {config.training.epochs}, Batch Size: {config.training.batch_size}")
        # TODO: Initialize dataset, dataloader, model, and engine here
    else:
        print("Starting inference mode")
        # TODO: Load weights and run predictions here

if __name__ == "__main__":
    main()
