# RFM
The code for the paper: "Feedback is important: response-aware feedback mechanism for background based conversation."

## Requirements 
* python 3.7
* pytorch 1.7.0

## Datasets 
- Download the [raw data version of Holl-E](https://github.com/nikitacs16/Holl-E), and put the raw data files (train_data.json, dev_data.json and test_data.json）in the directory `/dataset/raw_data`.
- Then, run the preprocessing script:
```
python Prepare_holl.py
```
- Download the `glove.6B.300d.txt` and put it in `/dataset/oracle` and `/dataset/mixed`.

## Run training, validation, and testing
To train or test your model, run:
```
python -m torch.distributed.launch --nproc_per_node=num_GPU Run_GLKS.py --mode='train/test'
```

## Addition
More descriptions will be released in a few days...
