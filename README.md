# Learning_to_diversify
This is the official code repository for ICCV2021 'Learning to Diversify for Single Domain Generalization'. 

Paper Link: http://arxiv.org/abs/2108.11726

## Update: Single DG with Resnet-18
Recently, we receive increasing enquiry about single DG on PACS with Resnet-18 Backbone. (In the paper, we reported Alexnet result)
Please try hyperparameters lr=0.002 and e=50, to start your experiment. 

We report the following single DG result on PACS, with resnet-18 as the backbone network:

|Src. domain    | P       | A     | C     | S    |avg. |
|---             | ------- |-------|-------| -----| --- |
| Avg. Tar. Acc. | 52.29   | 76.91 | 77.88 | 53.66|65.18|


## Quick start: (Generalizing from art, cartoon, sketch to photo domain with ResNet-18)
1. Install the required packages (including HuggingFace `datasets`):
```
pip install datasets
```
2. Execute the following code. For the PACS task the dataset is now downloaded and
   cached automatically via HuggingFace `datasets` (`flwrlabs/pacs`, ~191MB on first
   run, cached under `~/.cache/huggingface`), so no manual download is needed:
```
bash run_main_PACS.sh
```

### Data backend
PACS is loaded through HuggingFace `datasets` by default (`--data_backend=auto`).
- `--data_backend=hf`: force HuggingFace loading (PACS).
- `--data_backend=files`: use the original local txt-list / image-folder pipeline
  (expects images under the paths hardcoded in `data/JigsawLoader.py`).

VLCS and Office-Home still use the local-file pipeline.

## Change dataset
In line 266-300 of train.py, we provide 3 different datasets settings (PACS, VLCS, OFFICE-HOME).
You can simply uncomment it to start your own experiment. It may require hyper-parameter fine tuning for some of the tasks.


