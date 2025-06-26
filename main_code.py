from datasets import load_dataset
from transformers import ViTForImageClassification, ViTImageProcessor, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, random_split
import numpy as np

# Load the dataset
dataset = load_dataset("aryadytm/vehicle-classification")

# Split the training dataset into train and validation sets
train_size = int(0.8 * len(dataset["train"]))
val_size = len(dataset["train"]) - train_size
train_data, val_data = random_split(dataset["train"], [train_size, val_size])

# Load the image processor (replaces the deprecated ViTFeatureExtractor)
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Define a custom dataset class to apply preprocessing
class VehicleDataset(Dataset):
    def __init__(self, data, feature_extractor):
        self.data = data
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve the actual data instance from the underlying dataset
        item = self.data[idx] if isinstance(self.data, dict) else self.data.dataset[self.data.indices[idx]]
        image = item["image"]
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = {key: val.squeeze() for key, val in inputs.items()}
        inputs["label"] = torch.tensor(item["label"])
        return inputs

# Preprocess the dataset
train_dataset = VehicleDataset(train_data, feature_extractor)
val_dataset = VehicleDataset(val_data, feature_extractor)

# Define the model
# Define the model with ignore_mismatched_sizes set to True
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(dataset["train"].features["label"].names),
    ignore_mismatched_sizes=True
)


# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",

    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Evaluate the model
metrics = trainer.evaluate()
print(metrics)
