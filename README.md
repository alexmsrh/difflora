# DiffLoRA: Differential Low-Rank Adapters for Large Language Models

This repository contains the classes to load the differential low rank adapters on any model that follows the Llama model classes from transformers.


## Quickstart

```bash
pip install -r requirements.txt
```

Make sure to use transformers version 4.46.3 or prior.

An example config and code to initialize the differential attention modules are in `quickstart.ipynb`.

Then, the model can be trained and used for inference like a usual pytorch/transformers model. 


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
