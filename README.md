# RFM: response-aware feedback mechanism for background based conversation

The code for the paper [RFM: response-aware feedback mechanism for background based conversation](https://link.springer.com/article/10.1007/s10489-022-04056-4).

![RFM model figure](https://github.com/ChenchenJT/RFM/blob/main/figure/RFM.png)

## Reference

If you use any source code included in this repo in your work, please cite the following paper.

```text
@article{chen2022rfm,
  title={RFM: response-aware feedback mechanism for background based conversation},
  author={Chen, Jiatao and Zeng, Biqing and Du, Zhibin and Deng, Huimin and Xu, Mayi and Gan, Zibang and Ding, Meirong},
  journal={Applied Intelligence},
  year={2022},
  publisher={Springer},
  doi={10.1007/s10489-022-04056-4}
}
```

## Requirements

- python 3.7
- pytorch 1.7.0

## Datasets

- Download the [raw data version of Holl-E](https://github.com/nikitacs16/Holl-E), and put the raw data files (train_data.json, dev_data.json and test_data.json) in the directory `dataset/holl/raw_data/`.
- Then, run the preprocessing script:

```shell
python Prepare_holl.py
```

- As for Wizard of Wikipedia(WoW) dataset, we use the [DukeNet](https://github.com/ChuanMeng/DukeNet) version. Download the [Wizard of Wikipedia](https://drive.google.com/drive/folders/1zS0xRy-UgQTafNhxGBGS4in6zmAMKlVM) dataset, and put data files in the directory `dataset/wizard_of_wikipedia/`.
- Download the `glove.6B.300d.txt` and put it in `dataset/holl_e/oracle/`, `dataset/holl_e/mixed/` and `dataset/wizard_of_wikipedia/`.

## Run training, validation, and testing

To train or test your model, run:

```
# Holl-E dataset
python -m torch.distributed.launch --nproc_per_node=num_GPU Run_RFM_Holl.py --mode='train/test'

# Wizard of Wikipedia dataset
python -m torch.distributed.launch --nproc_per_node=num_GPU Run_RFM_WoW.py --mode='train/test'
```

If you want to run multiple references(MR) test version in Holl-E dataset, please add `--test='MR'` in the run script.

## Model checkpoint

We upload our model checkpoints on two datasets, you can manually download them at [here](https://drive.google.com/drive/folders/1ziTSrqN2bJD6KGA0sCXNOHk6plEhybmZ?usp=sharing).
