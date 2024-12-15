import copy
import os

import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

# HINT: Use this class mapping to merge similar classes and ignore classes that do not work very well
CLASS_MAPPING = {
    "Rock": "Rock",
    "Psych-Rock": "Rock",
    "Indie-Rock": None,
    "Post-Rock": "Rock",
    "Psych-Folk": "Folk",
    "Folk": "Folk",
    "Metal": "Metal",
    "Punk": "Metal",
    "Post-Punk": None,
    "Trip-Hop": "Trip-Hop",
    "Pop": "Pop",
    "Electronic": "Electronic",
    "Hip-Hop": "Hip-Hop",
    "Classical": "Classical",
    "Blues": "Blues",
    "Chiptune": "Electronic",
    "Jazz": "Jazz",
    "Soundtrack": None,
    "International": None,
    "Old-Time": None,
}


def torch_train_val_split(
    dataset, batch_train, batch_eval, val_size=0.2, shuffle=True, seed=420, test=False
):
    """
    Split a dataset into training, validation, and optionally test sets with PyTorch DataLoader.
    
    Args:
        dataset: PyTorch Dataset object
        batch_train: Batch size for training
        batch_eval: Batch size for validation/test
        val_size: Size of validation (and test if test=True) set as fraction of total dataset
        shuffle: Whether to shuffle the indices before splitting
        seed: Random seed for reproducibility
        test: If True, creates three splits (train/val/test) instead of two (train/val)
    
    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        test_loader: DataLoader for test set if test=True, else None
    """
    # Calculate dataset size and create indices
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    
    if test:
        # For test mode, we want three equal splits
        # If val_size is 0.2, we want:
        # - test_size = 0.1 (half of val_size)
        # - val_size = 0.1 (half of val_size)
        # - train_size = 0.8 (remaining portion)
        split_size = int(np.floor(val_size * dataset_size / 2))
        
        # Create the splits
        test_indices = indices[:split_size]  # First portion for test
        val_indices = indices[split_size:split_size * 2]  # Second portion for validation
        train_indices = indices[split_size * 2:]  # Remainder for training
    else:
        # For validation-only mode, we want two splits
        val_split = int(np.floor(val_size * dataset_size))
        val_indices = indices[:val_split]
        train_indices = indices[val_split:]
        test_indices = None
    
    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices) if test else None
    
    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=batch_train, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_eval, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_eval, sampler=test_sampler) if test else None
    
    return train_loader, val_loader, test_loader


def read_spectrogram(spectrogram_file, feat_type):
    spectrogram = np.load(spectrogram_file).astype(np.float32)
    # spectrograms contains a fused mel spectrogram and chromagram    
    if feat_type=='mel':
        return spectrogram[:128, :].T
    elif feat_type=='chroma':
        return spectrogram[128:, :].T

    return spectrogram.T


class LabelTransformer(LabelEncoder):
    def inverse(self, y):
        try:
            return super(LabelTransformer, self).inverse_transform(y)
        except:
            return super(LabelTransformer, self).inverse_transform([y])

    def transform(self, y):
        try:
            return super(LabelTransformer, self).transform(y)
        except:
            return super(LabelTransformer, self).transform([y])


class PaddingTransform(object):
    def __init__(self, max_length, padding_value=0):
        self.max_length = max_length
        self.padding_value = padding_value

    def __call__(self, s):
        if len(s) == self.max_length:
            return s

        if len(s) > self.max_length:
            return s[: self.max_length]

        if len(s) < self.max_length:
            s1 = copy.deepcopy(s)
            pad = np.zeros((self.max_length - s.shape[0], s.shape[1]), dtype=np.float32)
            s1 = np.vstack((s1, pad))
            return s1


class SpectrogramDataset(Dataset):
    def __init__(
        self, path, class_mapping=None, train=True, feat_type='mel', 
        max_length=-1, regression=None, multitask=False
    ):
        """
        Initialize dataset for either classification or regression tasks.
        
        Args:
            path: Path to the dataset directory
            class_mapping: Dictionary mapping raw class names to processed class names
            train: Boolean indicating if this is training data
            feat_type: Type of features to use ('mel', 'chroma', or 'fused')
            max_length: Maximum length of spectrograms (-1 for no limit)
            regression: Which column to use for regression (None for classification)
            multitask: Whether to return all regression targets
        """
        t = "train" if train else "test"
        p = os.path.join(path, t)
        self.regression = regression
        self.multitask = multitask
        self.feat_type = feat_type

        self.full_path = p
        self.index = os.path.join(path, "{}_labels.txt".format(t))
        
        print(f"Loading dataset from {self.full_path}")
        print(f"Using labels file: {self.index}")
        print(f"{'Multitask' if multitask else 'Single-task'} regression mode: {regression}")
        
        self.files, labels = self.get_files_labels(self.index, class_mapping)
        self.feats = [read_spectrogram(os.path.join(p, f), feat_type) for f in self.files]
        self.feat_dim = self.feats[0].shape[1]
        self.lengths = [len(i) for i in self.feats]
        self.max_length = max(self.lengths) if max_length <= 0 else max_length
        self.zero_pad_and_stack = PaddingTransform(self.max_length)
        
        # Handle labels based on task type
        if isinstance(labels, (list, tuple)):
            if not regression and not multitask:
                # Classification case
                self.label_transformer = LabelTransformer()
                self.labels = np.array(
                    self.label_transformer.fit_transform(labels)
                ).astype("int64")
            else:
                # Regression case (single or multitask)
                self.labels = np.array(labels).astype("float32")
                if self.multitask:
                    print(f"Multitask labels shape: {self.labels.shape}")
                else:
                    print(f"Single task labels shape: {self.labels.shape}")

    def get_files_labels(self, txt, class_mapping):
        """
        Read and process files and their corresponding labels.
        For regression tasks, expects a CSV format with columns: 
        filename, valence, energy, danceability
        """
        if not os.path.exists(txt):
            raise FileNotFoundError(f"Labels file not found: {txt}")
        
        with open(txt, "r") as fd:
            lines = [l.rstrip().split("\t") for l in fd.readlines()[1:]]
        
        print(f"Read {len(lines)} lines from labels file")
        
        files, labels = [], []
        for line_num, l in enumerate(lines, 1):
            try:
                if self.regression is not None or self.multitask:
                    # Handle regression case (single or multitask)
                    values = l[0].split(",")
                    if len(values) < 4:
                        print(f"Warning: Line {line_num} has insufficient values: {l}")
                        continue
                    
                    # Process filename
                    filename = values[0]
                    if not filename.endswith('.npy'):
                        filename += ".fused.full.npy"
                    
                    # Get regression targets
                    if self.multitask:
                        # For multitask, return all regression targets
                        regression_values = [float(x) for x in values[1:4]]
                        labels.append(regression_values)
                    else:
                        # For single task, return only the specified target
                        labels.append(float(values[self.regression]))
                    
                    files.append(filename)
                else:
                    # Handle classification case
                    if len(l) < 2:
                        print(f"Warning: Line {line_num} has wrong format: {l}")
                        continue
                    
                    label = l[1]
                    if class_mapping:
                        label = class_mapping[l[1]]
                    if not label:
                        continue
                    
                    fname = l[0]
                    if fname.endswith(".gz"):
                        fname = ".".join(fname.split(".")[:-1])
                    
                    # Handle special cases
                    if 'fma_genre_spectrograms_beat' in self.full_path:
                        fname = fname.replace('beatsync.fused', 'fused.full')            
                    if 'test' in self.full_path:
                        fname = fname.replace('full.fused', 'fused.full')
                    
                    files.append(fname)
                    labels.append(label)
                
            except Exception as e:
                print(f"Error processing line {line_num}: {str(e)}")
                continue
        
        print(f"Successfully processed {len(files)} valid entries")
        
        if len(files) == 0:
            raise ValueError("No valid files found in the labels file")
        
        return files, labels

    def __getitem__(self, item):
        """Get a single item from the dataset."""
        length = min(self.lengths[item], self.max_length)
        return self.zero_pad_and_stack(self.feats[item]), self.labels[item], length

    def __len__(self):
        """Get the total size of the dataset."""
        return len(self.labels)


if __name__ == "__main__":
    dataset = SpectrogramDataset(
        "data/fma_genre_spectrograms", class_mapping=CLASS_MAPPING, train=True, feat_type='chroma'
    )

    print(dataset[10])
    print(f"Input: {dataset[10][0].shape}")
    print(f"Label: {dataset[10][1]}")
    print(f"Original length: {dataset[10][2]}")
