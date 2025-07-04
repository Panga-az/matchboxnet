from transformers.configuration_utils import PretrainedConfig


class MatchboxNetConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a
    [`MatchboxNetForAudioClassification`] model. It is used to instantiate a MatchboxNet model according
    to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs,
    initialize the model, or save/load configuration files.

    Args:
        input_channels (`int`, *optional*, defaults to 64):
            Number of input channels, typically the number of MFCCs or filterbank features.
        num_classes (`int`, *optional*, defaults to 30):
            Number of target labels for classification.
        B (`int`, *optional*, defaults to 3):
            Number of residual blocks (MainBlocks).
        R (`int`, *optional*, defaults to 2):
            Number of sub-blocks per residual block.
        C (`int`, *optional*, defaults to 64):
            Number of internal channels in residual blocks.
        kernel_sizes (`List[int]`, *optional*):
            List of kernel sizes for each residual block. If `None`, defaults to ` [11 + 2*i for i in range(B)] B(number of blocks)`.
        target_sr (`int`, *optional*, defaults to 16000):
            Sampling rate expected for input audio.
        n_mfcc (`int`, *optional*, defaults to 64):
            Number of MFCCs to extract from raw waveform.
        fixed_length (`int`, *optional*, defaults to 128):
            Number of time steps expected after MFCC extraction and padding.
        
        padding_value(float): padding value.
        do_normalize (bool, optional, defaults to True): Whether to apply zero-mean unit-variance normalization.
        
        label2id (`Dict[str, int]`, *optional*):
            Dictionary mapping class names to IDs.
        id2label (`Dict[int, str]`, *optional*):
            Dictionary mapping IDs to class names.
        **kwargs:
            Additional keyword arguments passed to the `PretrainedConfig`.
    """

    model_type = "matchboxnet"
  

    def __init__(
        self,
        input_channels=64,
        num_classes=30,
        B=3,
        R=2,
        C=64,
        kernel_sizes=None,
        target_sr=16000,
        n_mfcc=64,
        fixed_length=128,
        do_normalize: bool = True,
        padding_value: float = 0.0,
        label2id=None,
        id2label=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.B = B
        self.R = R
        self.C = C
        self.kernel_sizes = kernel_sizes
        self.target_sr = target_sr
        self.n_mfcc = n_mfcc
        self.fixed_length = fixed_length
        self.do_normalize = do_normalize
        self.padding_value = padding_value

        if label2id is None or id2label is None:
            self.id2label = {i: str(i) for i in range(self.num_classes)}
            self.label2id = {v: k for k, v in self.id2label.items()}
        else:
            self.label2id = label2id
            self.id2label = id2label
            
__all__ = ["MatchboxNetConfig"]

