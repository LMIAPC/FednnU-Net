# FednnU-Net

### 🌟 Citation

If you find this work is helpful to your research, please cite two of our papers below:

```bibtex
@article{luo2023influence,
  title={Influence of data distribution on federated learning performance in tumor segmentation},
  author={Luo, Guibo and Liu, Tianyu and Lu, Jinghui and Chen, Xin and Yu, Lequan and Wu, Jian and Chen, Danny Z and Cai, Wenli},
  journal={Radiology: Artificial Intelligence},
  volume={5},
  number={3},
  pages={e220082},
  year={2023},
  publisher={Radiological Society of North America}
}
```

```bibtex
@article{liu2024fedfms,
  title={FedFMS: Exploring Federated Foundation Models for Medical Image Segmentation},
  author={Liu, Yuxi and Luo, Guibo and Zhu, Yuesheng},
  journal={arXiv preprint arXiv:2403.05408},
  year={2024}
}
```

### Prerequisites and Dataset Preparation

We implemented federated learning for 2D medical imaging using nnU-Net. The reference nnU-Net code can be found at: [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)

First, set up the dataset and environment according to the steps in the nnU-Net code.

The data.json file can be directly modified from this:

> {
> 
>  "channel_names": {  
> 
>     "0": "R", "1": "G", "2": "B"
> 
>  },
> 
>  "labels": {
> 
>    "background": 0,
> 
>    "brain tumour": 1
> 
>  },
> 
>  "numTraining": 36678,
> 
>  "file_ending": ".png"
> 
>  }

imagesTr and labelsTr contain the training set (thus the datasets for the clients that need training), and imagesTs and labelsTs contain the test set (e.g., datasets for unseen sites).

The training set should be strictly prefixed with client{}.format(i) to distinguish between different client datasets. For example, the images for the first client would be client0xxxx.png, and for the second client, they would be client1xxx.png.

Note that, for instance, if there are 4 clients and each one takes turns being the unseen site, you need to prepare 4 datasets, such as Dataset041_FeTS, Dataset042_FeTS, Dataset043_FeTS, Dataset044_FeTS. Each dataset’s imagesTs folder contains the unseen site data, and imagesTr contains the visible data from the other three sites. Complete the entire data preprocessing, training, inference, and testing process sequentially for each.

Example structure:

> Dataset041_FeTS
> 
> -imagesTr
> 
> --client1FeTS2022_00105.nii.gz5_0000.png
> 
> --client2FeTS2022_00470.nii.gz129_0000.png
> 
> --....
> 
> -labelsTr
> 
> --client1FeTS2022_00105.nii.gz5.png
> 
> --client2FeTS2022_00470.nii.gz129.png
> 
> -imagesTs
> 
> --client0FeTS2022_00131.nii.gz5_0000.png
> 
> --...
> 
> -labelsTs
> 
> --client0FeTS2022_00131.nii.gz5.png
> 
> --...

### Dataset Preprocessing

Dataset Preprocessing

```shell
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrit
```

`DATASET_ID` is the ID you set in datasetname during the data conversion step. For example, Dataset041_FeTS would be 41:

```shell
nnUNetv2_plan_and_preprocess -d 41 --verify_dataset_integrity
```

### Training

```bash
CUDA_VISIBLE_DEVICES=1 python nnunetv2/run/run_training.py 41 2d 5 --unseen_site 0 --client_num 4
```

Here, 41 is the dataset ID, 2d is the type, 5 is the number of folds for training (only train once, with 80% of each client’s dataset used for training and 20% for validation), --unseen_site is the federated learning test site, and --client_num is the total number of clients in federated learning (including the unseen one).

### Inference

This step generates prediction masks from the test set and saves them as PNG files:

```bash
CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i /mnt/diskB/lyx/nnUnet_data/prostate/nnUNet_raw/nnUNet_raw_data/Dataset042_FeTS/imagesTs -o /mnt/diskB/lyx/nnUnet_data/prostate/nnUNet_raw/nnUNet_raw_data/Dataset042_FeTS/output -d 42 -c 2d -f 5
```

You only need to change -i, -o, and -d to the corresponding test dataset, mask output folder, and dataset ID.

### Testing and Evaluation

Calculate dice scores and other results based on the masks output during the inference process:

```bash
python ./nnunetv2/evaluation/evaluate_predictions.py
```

Modify the paths in this Python file to correspond to the test folder paths:

```python
    folder_ref = '/mnt/diskB/lyx/nnUnet_data/prostate/nnUNet_raw/nnUNet_raw_data/Dataset041_FeTS/labelsTs'
    folder_pred = '/mnt/diskB/lyx/nnUnet_data/prostate/nnUNet_raw/nnUNet_raw_data/Dataset041_FeTS/output'
    output_file = '/mnt/diskB/lyx/nnUnet_data/prostate/nnUNet_raw/nnUNet_raw_data/Dataset041_FeTS/summary.json'
```

output_file = '/mnt/diskB/lyx/nnUnet_data/prostate/nnUNet_raw/nnUNet_raw_data/Dataset041_FeTS/summary.json' is where the dice scores and other metrics will be output.
