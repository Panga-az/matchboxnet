import torch
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import PreTrainedModel
import torch.nn as nn
from .utils import MatchboxNet 
from .config import MatchboxNetConfig

class MatchboxNetForAudioClassification(PreTrainedModel):
    r"""
    MatchboxNet model for audio classification tasks compatible with Hugging Face Transformers.

    This wraps a MatchboxNet encoder and adds classification logic using a cross-entropy loss.

    For more details, see:
    https://arxiv.org/pdf/2004.08531

    Args:
        config (MatchboxNetConfig): Model configuration class.
    """
    
    config_class = MatchboxNetConfig
    base_model_prefix = "matchboxnet"
    
    def __init__(
        self,
        config: MatchboxNetConfig,
    ):
        super().__init__(config=config)
        self.config = config
      
        # Architecture MatchboxNet complète
        self.matchboxnet = MatchboxNet(
            B=config.B,
            R=config.R,
            C=config.C,
            kernel_sizes=config.kernel_sizes,
            num_classes=config.num_classes,
            input_channels=config.input_channels,
        )
        self.loss_fn = nn.CrossEntropyLoss()
        
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_values=None, labels=None, **kwargs):
        """
        Forward pass for audio classification.

        Args:
            input_values (torch.FloatTensor): Input tensor of shape (batch_size, input_channels, time_steps).
            labels (torch.LongTensor, optional): Ground truth labels for classification.

        Returns:
            transformers.modeling_outputs.SequenceClassifierOutput: A dataclass containing:
                - loss (optional): Cross-entropy loss if labels provided.
                - logits (torch.FloatTensor): Prediction logits of shape (batch_size, num_classes).
        """
        logits = self.matchboxnet(input_values)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        
        return SequenceClassifierOutput(loss=loss, logits=logits)
    
    
__all__ = [
    "MatchboxNetForAudioClassification",
    "MatchboxNet"

]