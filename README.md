# MatchboxNet

PyTorch + Hugging Face Transformers implementation of [MatchboxNet](https://arxiv.org/abs/2004.08531) for audio classification tasks.

## Table of Contents


- [Installation](#installation)
- [Usage](#usage)
  - [Load Datasets](#load-datasets)
  - [Config the model](#Configthe-model)
  - [Prepare datasets for training](#Prepare-datasets-for-training)
  - [Using the model to train](#using-the-model-to-train)
- [Running Tests](#running-tests)
- [Contributing](#contributing)
- [Citation](#citation)



### Installation

```bash
pip install matchboxnet
```

## Usage

To use matchboxenet package for audio classification tasks, you have to : 



### Load Datatesets
If your dataset is hosted on the Hugging Face Hub and follows the `audio-classification` task format

```python
from datasets import load_dataset

ds = load_dataset('your dataset')

ds_train = ds["train"]          
ds_val   = ds["validation"]     
ds_test  = ds["test"] 
```

### Config the model.

Uses `MatchboxNetConfig`, which contains all the necessary parameters for the model.

```python
from matchboxnet.config import MatchboxNetConfig

labels = ds_train.features["label"].names 


label2id = {lab: i for i, lab in enumerate(labels)}
id2label = {i: lab for lab, i in label2id.items()}

config = MatchboxNetConfig(
    input_channels=64,
    target_sr=16_000,
    n_mfcc=64,
    num_classes = len(labels),
    fixed_length=128,
    do_normalize=True,
    B=3,
    R=2,
    C=64,
    paddig_value = 0.0,
    label2id = label2id ,
    id2label = id2label
)
```

### Prepare datasets for training

Use the `MatchboxNetDataset` to prepare your Hugging face dataset for training:

```python
import torch
from matchboxnet.config import MatchboxNetConfig,
from matchboxnet.dataset import MatchboxNetDataset         


train_ds = MatchboxNetDataset(ds_train, config=config, augment=True)
eval_ds = MatchboxNetDataset(ds_val, config=config, augment=False)
train_ds = MatchboxNetDataset(ds_train, config=config, augment=False)
```


### Using the model to train

```python
from matchboxnet.model import MatchboxNetForAudioClassification

# Compute metrics
import evaluate

accuracy_metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return accuracy_metric.compute(predictions=preds, references=labels)


model = MatchboxNetForAudioClassification(config=config)

from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

training_args = TrainingArguments(...)

#Just an example
trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=train_ds,
  eval_dataset=eval_ds,
  compute_metrics=compute_metrics,
  callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

```

## Contributing

All contributions are welcome!

## License

[MIT License](LICENSE)

## Citation

If you use this codebase or MatchboxNet in your work, please cite the original paper:

```bibtex
@inproceedings{Majumdar_2020, 
  series={interspeech_2020},
  title={MatchboxNet: 1D Time-Channel Separable Convolutional Neural Network Architecture for Speech Commands Recognition},
  url={http://dx.doi.org/10.21437/Interspeech.2020-1058},
  DOI={10.21437/interspeech.2020-1058},
  booktitle={Interspeech 2020},
  publisher={ISCA},
  author={Majumdar, Somshubra and Ginsburg, Boris},
  year={2020},
  month=oct,
  collection={interspeech_2020}
}
```

