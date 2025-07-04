
import torch
import torchaudio
import numpy as np
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import logging

from .utils import symmetric_pad_or_truncate

logger = logging.get_logger(__name__)


class MatchboxNetFeatureExtractor(SequenceFeatureExtractor):
    
    """
    Feature extractor for MatchboxNet audio classification models.

    Implements feature preparation following the MatchboxNet design ( https://arxiv.org/pdf/2004.08531):
    - Extraction of MFCC features
    - Symmetric padding/truncation

    Link to original MatchboxNet paper:  https://arxiv.org/pdf/2004.08531

    The implementation covers:
        * MFCC computation (25 ms windows, 10 ms hop, 64 coefficients)
        * Fixed-length framing (128 frames)
        
    

    Example:

        >>> from matchboxnet.feature_extraction import MatchboxNetFeatureExtractor
        >>> import numpy as np
        >>> feat_ext = MatchboxNetFeatureExtractor()
        >>> raw_audio = np.random.randn(16000)  # 1-second audio at 16kHz
        >>> batch = feat_ext(raw_audio)
        >>> input_values = batch["input_values"]  # shape (1, 64, 128)

    Args:
        sampling_rate (int, optional, defaults to 16000): Sampling rate of input audio.
        n_mfcc (int, optional, defaults to 64): Number of MFCC coefficients to extract.
        fixed_length (int, optional, defaults to 128): Target number of time frames after padding/truncation.
        **kwargs: Additional keyword arguments forwarded to SequenceFeatureExtractor.

        Attributes:
        mfcc_transform (torchaudio.transforms.MFCC): MFCC computation transform.
    """
    model_input_names = ["input_values"]

    def __init__(
        self,
        sampling_rate: int = 16000,
        n_mfcc: int = 64,
        fixed_length: int = 128,
        padding_value: float = 0.0,
        do_normalize : bool =  True,
        **kwargs
    ):
        super().__init__(feature_size=n_mfcc, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.sampling_rate = sampling_rate
        self.n_mfcc = n_mfcc
        self.fixed_length = fixed_length
        self.padding_value = padding_value
        self.do_normalize = do_normalize
        self.melkwargs =  {
            "n_fft": int(0.025 * sampling_rate),
            "hop_length": int(0.010 * sampling_rate),
            "n_mels": n_mfcc,
        }
        self._mfcc_args = {
            "sample_rate": self.sampling_rate,
            "n_mfcc": self.n_mfcc,
            "melkwargs": self.melkwargs,
        }
        self._build_mfcc()

    def _build_mfcc(self):
        """Build torchaudio MFCC from saved args"""
        self.mfcc_transform = torchaudio.transforms.MFCC(**self._mfcc_args)

    def to_dict(self) -> dict:
        """
        Serialize only JSON-serializable fields, dropping transform objects.
        """
        return {
            "feature_extractor_type": self.__class__.__name__,
            "feature_size": self.n_mfcc,
            "fixed_length": self.fixed_length,
            "melkwargs": self.melkwargs,
            "n_mfcc": self.n_mfcc,
            "padding_value": self.padding_value,
            "sampling_rate": self.sampling_rate,
        }

    @classmethod
    def from_dict(cls, data: dict):
        """
        Recreate extractor from JSON schema.
        """
        return cls(
            sampling_rate=data.get("sampling_rate"),
            n_mfcc=data.get("n_mfcc"),
            fixed_length=data.get("fixed_length"),
            padding_value=data.get("padding_value"),
            melkwargs=data.get("melkwargs"),
        )

    @staticmethod
    def zero_mean_unit_var_norm(inputs: list[np.ndarray], padding_value: float = 0.0) -> list[np.ndarray]:
        """
        Apply zero-mean, unit-variance normalization to feature arrays.
        """
        normed = []
        for x in inputs:
            m, v = x.mean(), x.var()
            x_norm = (x - m) / np.sqrt(v + 1e-7)
            if padding_value != 0.0:
                x_norm[x == padding_value] = padding_value
            normed.append(x_norm)
        return normed

    def __call__(
        self,
        raw_speech,
        return_tensors: str = "pt",
        **kwargs
    ) -> BatchFeature:
        """
        Process raw audio into MFCC inputs (no augment).
        """
        arrays, paths = [], []
        if isinstance(raw_speech, str):
            paths = [raw_speech]
        elif isinstance(raw_speech, list) and all(isinstance(x, str) for x in raw_speech):
            paths = raw_speech
        elif isinstance(raw_speech, np.ndarray):
            arrays = [raw_speech]
        elif isinstance(raw_speech, list) and all(isinstance(x, np.ndarray) for x in raw_speech):
            arrays = raw_speech
        else:
            raise ValueError(f"Unsupported raw_speech type: {type(raw_speech)}")

        for path in paths:
            wav, sr = torchaudio.load(path)
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != self.sampling_rate:
                wav = torchaudio.transforms.Resample(sr, self.sampling_rate)(wav)
            arrays.append(wav.squeeze(0).numpy())

        features = []
        for arr in arrays:
            t = torch.as_tensor(arr, dtype=torch.float32)
            mfcc = self.mfcc_transform(t).squeeze(0)
            mfcc = symmetric_pad_or_truncate(mfcc, self.fixed_length)
            features.append(mfcc)
        
        if self.do_normalize:
            features_np = [f.numpy() for f in features]
            features_np = self.zero_mean_unit_var_norm(features_np, padding_value=self.padding_value)
            features = [torch.tensor(f, dtype=torch.float32) for f in features_np]
            
        batch = torch.stack(features, dim=0)
        return BatchFeature({"input_values": batch}) if return_tensors == "pt" else BatchFeature({"input_values": batch.numpy()})

__all__ = ["MatchboxNetFeatureExtractor"]