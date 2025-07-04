from torch.utils.data import Dataset
import numpy as np 
import torch
import torchaudio
from .feature_extraction import MatchboxNetFeatureExtractor
from .config import MatchboxNetConfig

class MatchboxNetDataset(Dataset):
    r"""
    PyTorch Dataset that yields MFCC feature tensors and corresponding labels,
    with optional on-the-fly audio augmentations.

    This class uses a base FeatureExtractor to compute MFCCs without augmentations,
    then applies time shift, white noise, SpecAugment, and SpecCutout
    if `augment=True`.
    """
    def __init__(
        self,
        dataset,
        config: MatchboxNetConfig,
        audio_column: str = "audio",
        label_column: str = "label",
        time_shift_ms: int = 5,
        augment : bool = True,
        noise_db: tuple[float, float] = (-90, -46),
        spec_time_masks: int = 2,
        spec_time_max: int = 25,
        spec_freq_masks: int = 2,
        spec_freq_max: int = 15,
        spec_cutout_masks: int = 5,
        **kwargs
    ):
        r"""
        Initialize the dataset.

        Args:
            dataset (Dataset): Source dataset providing dicts with audio arrays and labels.
            config (MatchboxNetConfig): Config.
            audio_column (str): Key for the audio array in each example. Defaults to "audio".
            label_column (str): Key for the label in each example. Defaults to "label".
            time_shift_ms (int): Maximum time shift in milliseconds. Defaults to 5.
            augment (bool, optional, defaults to True): Whether to apply augmentations.
            noise_db (tuple): dB range for white noise injection. Defaults to (-90, -46).
            spec_time_masks (int): Number of time masks for SpecAugment. Defaults to 2.
            spec_time_max (int): Maximum width of time masks. Defaults to 25 frames.
            spec_freq_masks (int): Number of frequency masks. Defaults to 2.
            spec_freq_max (int): Maximum width of frequency masks. Defaults to 15 bins.
            spec_cutout_masks (int): Number of rectangular cutout masks. Defaults to 5.
        """
        self.dataset = dataset
        self.config = config
        self.audio_column = audio_column
        self.label_column = label_column
        self.augment = augment
        
        self.fe : MatchboxNetFeatureExtractor = MatchboxNetFeatureExtractor(
            sampling_rate=self.config.target_sr,
            n_mfcc=self.config.n_mfcc,
            fixed_length=self.config.fixed_length,
            padding_value=self.config.padding_value,
            do_normalize=False
        )

        
        # Augmentation parameters
        self.time_shift_ms = time_shift_ms
        self.noise_db = noise_db
        self.spec_time_masks = spec_time_masks
        self.spec_time_max = spec_time_max
        self.spec_freq_masks = spec_freq_masks
        self.spec_freq_max = spec_freq_max
        self.spec_cutout_masks = spec_cutout_masks

    def __len__(self) -> int:
        r"""
        Returns:
            int: Number of examples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        r"""
      
        Fetch and process a single example.

        Steps:
            1. Load raw audio array from `audio_column`.
            2. Use `feature_extractor` to compute raw MFCCs.
            3. If `augment=True`, apply:
                - Time shift
                - White noise
                - SpecAugment (time/freq masking)
                - SpecCutout (rectangular masks)
            4. Retrieve label and convert to tensor.

        Args:
            idx (int): Index of the example.

        Returns:
            dict: A dictionary with:
                - "input_values": Tensor of shape (n_mfcc, fixed_length)
                - "labels": LongTensor with the class label
        """

        # 1) Load audio waveform
        item = self.dataset[idx]
        signal = item[self.audio_column]["array"]

        # 2) Compute raw MFCC (no augmentation)
        batch = self.fe(signal)
        mfcc = batch["input_values"][0]

        if self.augment:
            # 3a) Time shift
            max_shift = int(self.time_shift_ms / 1000 * self.fe.fixed_length)
            shift = np.random.randint(-max_shift, max_shift) if max_shift > 0 else 0
            mfcc = torch.roll(mfcc, shifts=shift, dims=-1)

            # 3b) White noise
            db = float(np.random.uniform(*self.noise_db))
            mfcc = mfcc + torch.randn_like(mfcc) * (10 ** (db / 20))

            # 3c) SpecAugment: time and frequency masking
            for _ in range(self.spec_time_masks):
                mfcc = torchaudio.transforms.TimeMasking(time_mask_param=self.spec_time_max)(mfcc)
            for _ in range(self.spec_freq_masks):
                mfcc = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.spec_freq_max)(mfcc)

            # 3d) SpecCutout: rectangular masks
            for _ in range(self.spec_cutout_masks):
                t0 = np.random.randint(0, self.fe.fixed_length - self.spec_time_max)
                f0 = np.random.randint(0, self.fe.n_mfcc - self.spec_freq_max)
                mfcc[f0 : f0 + self.spec_freq_max, t0 : t0 + self.spec_time_max] = 0

        if self.config.do_normalize : 
            mfcc_np = mfcc.numpy()
            normed = self.fe.zero_mean_unit_var_norm([mfcc_np], padding_value=self.fe.padding_value)[0]
            mfcc = torch.from_numpy(normed)
            
        # 4) Load label
        label = item[self.label_column]
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {"input_values": mfcc, "labels": label_tensor}
