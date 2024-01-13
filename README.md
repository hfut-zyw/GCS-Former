# GCS-Former: A framework using Gesture Contrastive Space for sEMG based Gesture Recognition
This repo contains the code of our latest paper: GCS-Former: A framework using Gesture Contrastive Space for sEMG based Gesture Recognition



## Requirements
#### 1.System and related driver versions
- Windows 10 Professional 21H2
- A CUDA compatible GPU (with a memory size of 12GB or greater) 
- CUDA v11.8.89
- cuDNN v8.9.0

#### 2.Deep learning framework version
- Python v3.10.9
- torch v2.0.0+cu118

#### 3.Other related libraries
Please use the provided requirements.txt file for installation

```
pip install -r requirements.txt
```



## Data preparing

#### 1.Ninapro DB1 & DB2

Firstly download the [Ninapro DB1](http://ninaweb.hevs.ch/data1) and [Ninapro DB2](http://ninaweb.hevs.ch/data2) datasets. And then extract data files from the zip files.   

Your directory tree should look like this:  

```
{Root}/data/ninapro
├── db1
|   |—— s1
|   |—— s2
|   |   ...
|   └── s27
|       |—— S27_A1_E1.mat
|       |—— S27_A1_E2.mat
|       └── S27_A1_E3.mat
└── db2
    |—— DB2_s1
    |—— DB2_s2
    |   ...
    └── DB2_s40
        |—— S40_E1_A1.mat
        |—— S40_E2_A1.mat
        └── S40_E3_A1.mat
```

We provide two script files (located in the ninapro_file_process folder)  for convert the mat files to txt files.  
After convertion, your directory tree should look like this:   

```
{Root}/data/ninapro
├── db1_processed
|   |—— s1
|   |—— s2
|   |   ...
|   └── s27
|       |—— emg.txt
|		|—— exercise.txt
|       |—— rerepetition.txt
|       └── restimulus.txt
└── db2_processed
    |—— DB2_s1
    |—— DB2_s2
    |   ...
    └── DB2_s40
        |—— emg.txt
        |—— exercise.txt
        |—— rerepetition.txt
        └── restimulus.txt
```

#### 2.OH-Sentence DB1

Get the [OH-Sentence DB1](https://github.com/ZhangJiangtao-0108/OH-Sentence_Dataset) (DB1 is a subset of OH-Sentence database) dataset.   

And then extract data files from the zip files.   

Your directory tree should look like this:

```
{Root}/data/oh_sentence
├── DB1a
|   |—— sub1
|   |—— sub2
|   |   ...
|   └── sub14
|       |—— emg
|       └── imu
└── DB1b
    |—— sub15
    |—— sub16
    |—— sub17
    |—— sub18
    └── sub19
        |—— emg
        └── imu
```



## Quick Start

#### Main results

- Results on Ninapro DB1&DB2

| Method              | Dataset     | Movements and subjects | Window length (150ms) | Window length (200ms) |
| ------------------- | ----------- | ---------------------- | --------------------- | --------------------- |
| Random forests      | Ninapro DB1 | 50,27                  | -                     | 75.32                 |
| AtzoriNet           | Ninapro DB1 | 50,27                  | -                     | 66.59                 |
| GengNet             | Ninapro DB1 | 52,27                  | -                     | 77.8                  |
| Krishnapriya et al. | Ninapro DB1 | 52,27                  | -                     | 78.95                 |
| Ours                | Ninapro DB1 | 52,27                  | **81.74**             | **82.77**             |
| AtzoriNet           | Ninapro DB2 | 50,40                  | 60.27                 | -                     |
| **GCS-Former**      | Ninapro DB2 | 49,40                  | **69.80**             | **71.86**             |

- Results on OH-Sentence DB1

| Method         | subjects 1-9           | subjects 1-9            | subjects 10-19         | subjects 10-19          | all subjects           | all subjects            |
| -------------- | ---------------------- | ----------------------- | ---------------------- | ----------------------- | ---------------------- | ----------------------- |
|                | average representation | sequence representation | average representation | sequence representation | average representation | sequence representation |
| Alexnet        | 36.74                  | -                       | 31.68                  | -                       | 34.08                  | -                       |
| Resnet18       | 42.79                  | -                       | 38.88                  | -                       | 40.73                  | -                       |
| LSTM           | -                      | 11.96                   | -                      | 14.52                   | -                      | 13.31                   |
| Transformer    | 13.07                  | 13.04                   | 16.97                  | 16.47                   | 14.86                  | 14.85                   |
| **GCS-Former** | **81.18**              | **80.71**               | **80.52**              | **58.78**               | **80.94**              | **69.08**               |




#### Training and Evaluation

We encapsulated all the steps of dataset preprocessing, model building, model training, and model evaluation for each experiment into separate .py script files. You only need to run the .py file for each experiment to execute and obtain the results.

```
# Workspace : {Root}
cd experiment
```

```
python experiment1_db1_200ms.py

python experiment1_db1_150ms.py

python experiment1_db2_200ms.py

python experiment1_db2_150ms.py

python experiment2_db1.py

python experiment2_db2.py

python experiment3_db1.py

python experiment3_db2.py

python experiment4_db1_withoutgcs.py

python experiment4_db2_withougcs.py

python experiment4_db1_AlexnetEncoder.py

python experiment4_db2_AlexnetEncoder.py

python experiment5_GCSFormer_average_representation.py

python experiment5_GCSFormer_sequence_representation.py

python experiment5_AlexNet.py

python experiment5_Resnet18.py

python experiment5_LSTM_sequence_last.py

python experiment5_Transformer_average_representation.py

python experiment5_Transformer_sequence_representation.py
```



## Reference
```
...
```