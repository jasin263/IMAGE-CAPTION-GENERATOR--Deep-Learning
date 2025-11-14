#!/usr/bin/env python3
"""
Training script for custom BLIP image captioning model
Usage: python train_script.py --csv_path dataset.csv --image_dir images/ --epochs 10
"""

import argparse
from captioner.train_model import train_model

def main():
    parser = argparse.ArgumentParser(description='Train BLIP model on custom dataset')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to CSV file with image_path and caption columns')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='trained_model',
                       help='Directory to save trained model')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate')

    args = parser.parse_args()

    print("Starting training...")
    print(f"Dataset: {args.csv_path}")
    print(f"Images: {args.image_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("-" * 50)

    try:
        model, processor = train_model(
            csv_path=args.csv_path,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed: {str(e)}")

if __name__ == "__main__":
    main()
