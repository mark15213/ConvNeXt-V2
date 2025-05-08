import os
import time # For logging duration
from PIL import Image
from torch.utils.data import Dataset
# torchvision.transforms is usually imported in the main script

class WsiDatasetTxt(Dataset):
    def __init__(self, txt_file_path, transform=None, data_root=None):
        """
        Args:
            txt_file_path (string): Path to the txt file with image paths.
                                    It's highly recommended this file is pre-cleaned
                                    to contain only valid, existing image paths.
            transform (callable, optional): Optional transform to be applied on a sample.
            data_root (string, optional): A root directory to prepend to paths in txt_file
                                          if they are relative. If paths in txt_file are
                                          absolute, this should be None.
        """
        self.transform = transform
        self.data_root = data_root # Store for constructing full paths
        self.txt_file_path = txt_file_path
        self.image_paths = []
        self.labels = []  # Dummy labels for pre-training/self-supervised

        print(f"[WsiDatasetTxt INFO] Initializing dataset from: {self.txt_file_path}")
        if self.data_root:
            print(f"[WsiDatasetTxt INFO] Using data_root: {self.data_root} to prepend to relative paths.")

        start_time = time.time()
        print(f"[WsiDatasetTxt INFO] Starting to load image paths into memory. This can take a while for very large files...")

        loaded_count = 0
        skipped_empty_lines = 0

        try:
            with open(self.txt_file_path, 'r') as f:
                for i, line in enumerate(f):
                    img_filename_or_rel_path = line.strip()

                    if not img_filename_or_rel_path:
                        skipped_empty_lines += 1
                        continue

                    if self.data_root:
                        full_img_path = os.path.join(self.data_root, img_filename_or_rel_path)
                    else:
                        # Assuming paths in txt_file_path are already absolute
                        full_img_path = img_filename_or_rel_path
                    
                    self.image_paths.append(full_img_path)
                    self.labels.append(0)  # Assigning a dummy label 0
                    loaded_count += 1

                    if (loaded_count % 5000000 == 0) and loaded_count > 0: # Log every 5 million paths
                        elapsed_time = time.time() - start_time
                        print(f"[WsiDatasetTxt INFO] Loaded {loaded_count:,} image paths... ({elapsed_time:.2f}s elapsed)")
            
            end_time = time.time()
            if skipped_empty_lines > 0:
                print(f"[WsiDatasetTxt INFO] Skipped {skipped_empty_lines:,} empty lines in the text file.")
            print(f"[WsiDatasetTxt INFO] Successfully loaded {len(self.image_paths):,} image paths in {end_time - start_time:.2f} seconds.")

        except FileNotFoundError:
            print(f"[WsiDatasetTxt ERROR] Text file not found at {self.txt_file_path}")
            raise
        except Exception as e:
            print(f"[WsiDatasetTxt ERROR] Error reading or processing text file {self.txt_file_path}: {e}")
            raise

        if not self.image_paths:
            print(f"[WsiDatasetTxt WARNING] Found 0 image paths after processing {self.txt_file_path}. Check file content and `data_root`.")
            raise RuntimeError(f"No image paths loaded from {self.txt_file_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if idx >= len(self.image_paths): # Should not happen with a correct DataLoader setup
            # This check is more for direct indexing issues, DataLoader handles out-of-bounds from __len__
            print(f"[WsiDatasetTxt ERROR] Index {idx} is out of bounds for dataset length {len(self.image_paths)}.")
            # Returning None will be caught by the collate_fn.
            # Raising an error here might be too abrupt if it's a rare Dataloader glitch.
            return None, None # (image, label)

        img_path = self.image_paths[idx]
        # label = self.labels[idx] # Using 0 directly for simplicity as all labels are 0

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"[WsiDatasetTxt WARNING] GetItem: Image file not found at '{img_path}'. Path will be skipped by collate_fn.")
            return None, 0 # Return (None, dummy_label)
        except Exception as e: # Catches other PIL errors (corrupt image, etc.)
            print(f"[WsiDatasetTxt WARNING] GetItem: Failed to load image '{img_path}' due to: {e}. Path will be skipped by collate_fn.")
            return None, 0 # Return (None, dummy_label)

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"[WsiDatasetTxt WARNING] GetItem: Error applying transform to image '{img_path}': {e}. Path will be skipped by collate_fn.")
                return None, 0 # Return (None, dummy_label)
        
        return image, 0 # Return (transformed_image, dummy_label 0)