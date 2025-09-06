# DiffLoRA: Differential Low-Rank Adapters for Large Language Models

This repository contains the classes to load the differential low rank adapters on any model that follows the Llama model classes from transformers.


## Quickstart

Create a conda env with python 3.10. Then install requirements

```bash
pip install -r requirements.txt
```

Make sure to use transformers version 4.46.3 or prior as there were important refactorings in transormers after this version.

An example config and code to initialize the differential attention modules are in `quickstart.ipynb`.

## Train

Then, the model can be trained and used for inference like a usual pytorch/transformers model. The scripts we used are provided in `train/tulu/scripts`.

To use flash attention:
```bash
pip install flex-head-fa --no-build-isolation
```

If you wish to train the model using the provided training scripts, you may need to install open-instruct and olmes:
```bash
git clone https://github.com/allenai/open-instruct.git
cd open-instruct
pip install -e .

git clone https://github.com/allenai/olmes.git
cd olmes
pip install -e .
```

## Cite
```
@misc{
anonymous2025difflora,
title={DiffLo{RA}: Differential Low-Rank Adapters for Large Language Models},
author={Anonymous},
year={2025},
url={https://openreview.net/forum?id=7GLpLHIbvb}
}
```
