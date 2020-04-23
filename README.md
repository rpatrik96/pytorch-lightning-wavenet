## WaveNet in PyTorch
### Author: Patrik Reizinger (MIT License)
This is an implementation of WaveNet in PyTorch using PyTorch Lightning.

Although there are several implementation, those are quite old. Thus, I have written a concise and clean version, which is well documented.

## Training
For training, you need the `PennTreeBank` dataset, for which there are two wrapper classes in `penn_dataset.py`.

As PyTorch Lightning provides a super easy-to-use interface; thus, you only need to run the following:
```python
python wavenet_lightning.py
```