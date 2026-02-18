from dataset import RAVDESSDataset
from torch.utils.data import DataLoader

# Initialize Dataset (Use augment=False for speed check)
ds = RAVDESSDataset("ravdess_metadata.csv", augment=False)

# Create a DataLoader (Simulates training batch)
# batch_size=8 means we feed 8 images at a time
loader = DataLoader(ds, batch_size=8, shuffle=True)

# Grab one batch
images, labels = next(iter(loader))

print(f"Batch Shape: {images.shape}")
print(f"Labels: {labels}")

# EXPECTED OUTPUT:
# Batch Shape: torch.Size([8, 1, 128, 130])
# Labels: tensor([4, 2, 0, ...]) (random integers 0-7)

if images.shape[1:] == (1, 128, 130):
    print("\nSUCCESS: Data shape is correct for 2D CNN!")
else:
    print(f"\nWARNING: Unexpected shape {images.shape}. Check your duration/sr.")
    