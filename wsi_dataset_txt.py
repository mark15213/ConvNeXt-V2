import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class WsiDatasetTxt(Dataset):
    def __init__(self, txt_file_path, transform=None, data_root=None):
        """
        Args:
            txt_file_path (string): Path to the txt file with image paths.
            transform (callable, optional): Optional transform to be applied on a sample.
            data_root (string, optional): A root directory to prepend to paths in txt_file if they are relative.
                                          If paths in txt_file are absolute, this can be None.
        """
        self.transform = transform
        self.data_root = data_root
        self.image_paths = []
        self.labels = [] # For pre-training, we might not need actual labels, or use a dummy one

        try:
            with open(txt_file_path, 'r') as f:
                for line in f:
                    img_path = line.strip()
                    if not img_path: # Skip empty lines
                        continue

                    if self.data_root:
                        img_path = os.path.join(self.data_root, img_path)

                    if not os.path.exists(img_path):
                        print(f"Warning: Image path not found and will be skipped: {img_path}")
                        continue
                    
                    self.image_paths.append(img_path)
                    self.labels.append(0) # Assigning a dummy label 0

        except FileNotFoundError:
            raise RuntimeError(f"Error: Text file not found at {txt_file_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading or processing text file {txt_file_path}: {e}")

        if not self.image_paths:
            raise RuntimeError(f"Found 0 valid image paths in {txt_file_path}")

        print(f"Successfully loaded {len(self.image_paths)} image paths from {txt_file_path}")


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Skipping.")
            if idx + 1 < len(self.image_paths): # A simple attempt to get next if available
                print(f"Attempting to load next image instead of {img_path}")
                return self.__getitem__((idx + 1) % len(self.image_paths)) # Modulo to wrap around
            else:
                raise IOError(f"Could not load image {img_path} and no subsequent image available.")


        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx] # Return the dummy label

        return image, label