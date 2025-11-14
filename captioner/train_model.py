import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

class ImageCaptionDataset(Dataset):
    def __init__(self, dataframe, processor, max_length=50):
        self.dataframe = dataframe
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['image_path']
        caption = row['caption']

        # Load and process image
        image = Image.open(image_path).convert('RGB')
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        # Process text
        text_inputs = self.processor.tokenizer(
            caption,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )

        return {
            'pixel_values': pixel_values,
            'input_ids': text_inputs['input_ids'].squeeze(),
            'attention_mask': text_inputs['attention_mask'].squeeze()
        }

def train_model(csv_path, image_dir, output_dir="trained_model", epochs=5, batch_size=8, learning_rate=5e-5):
    """
    Train BLIP model on custom dataset

    Args:
        csv_path (str): Path to CSV file with columns 'image_path' and 'caption'
        image_dir (str): Directory containing images
        output_dir (str): Directory to save trained model
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate
    """

    # Load dataset
    df = pd.read_csv(csv_path)

    # Ensure image paths are absolute
    df['image_path'] = df['image_path'].apply(lambda x: os.path.join(image_dir, x) if not os.path.isabs(x) else x)

    # Split dataset
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Initialize processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Create datasets
    train_dataset = ImageCaptionDataset(train_df, processor)
    val_dataset = ImageCaptionDataset(val_df, processor)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass
            outputs = model(pixel_values=pixel_values,
                          labels=input_ids,
                          attention_mask=attention_mask)

            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = model(pixel_values=pixel_values,
                              labels=input_ids,
                              attention_mask=attention_mask)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        model.train()

    # Save trained model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    print(f"Model saved to {output_dir}")
    return model, processor

def generate_caption_with_trained_model(image_path, model_path="trained_model"):
    """
    Generate caption using trained model

    Args:
        image_path (str): Path to image
        model_path (str): Path to trained model directory

    Returns:
        str: Generated caption
    """
    try:
        # Load trained model
        processor = BlipProcessor.from_pretrained(model_path)
        model = BlipForConditionalGeneration.from_pretrained(model_path)

        # Load and process image
        image = Image.open(image_path).convert('RGB')
        inputs = processor(image, return_tensors="pt")

        # Generate caption
        with torch.no_grad():
            out = model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)

        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Example usage
    # train_model("captions.csv", "images/", epochs=10)
    pass
