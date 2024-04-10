# TSformer-SA

The official implementation of **[A Temporal-Spectral Fusion Transformer with Subject-specific Adapter for Enhancing RSVP-BCI Decoding](https://arxiv.org/abs/2401.06340)**.


![alt text](figure/Model1_revision.png)

TSformer-SA is a symmetrical dual-stream Transformer comprising a feature extractor, a cross-view interaction module, a fusion module, and a subject-specific adapter. The inputs consist of EEG temporal signals representing the temporal view and the spectrogram images representing the spectral view. The feature extractor tokenizes the inputs and extracts the view-specific features. Subsequently, the cross-view interaction module extracts the common features from both views and the fusion module fuses the two-view features for classification. The above three modules are trained during the pre-training stage and only the subject-specific adapter is trained in the fine-tuning stage.


## Installation

Follow the steps below to prepare the virtual environment.

Create and activate the environment:
```shell
conda create -n tsformer python=3.9
conda activate tsformer
pip install -r requirements.txt
```

## Train
```bash
python -m torch.distributed.launch --master_port 29502 --nproc_per_node=2 /TSformer-SA/Pre_train.py
python -m torch.distributed.launch --master_port 29502 --nproc_per_node=2 /TSformer-SA/Fine_tune.py
```



## Cite

If you find this code or our TSformer-SA paper helpful for your research, please cite our paper:

```bibtex
@article{li2024temporal,
  title={A Temporal-Spectral Fusion Transformer with Subject-specific Adapter for Enhancing RSVP-BCI Decoding},
  author={Li, Xujin and Wei, Wei and Qiu, Shuang and He, Huiguang},
  journal={arXiv preprint arXiv:2401.06340},
  year={2024}
}
```
