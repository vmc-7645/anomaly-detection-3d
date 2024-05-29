# Anomaly Detection 3D

An implementation of the ["Anomaly Detection in 3D Point Clouds using Deep Geometric Descriptors"](https://arxiv.org/pdf/2202.11660) paper in Python without pre-training with the ModelNet10 dataset and without generating the synthetic data.

## Training
After downloading the repo, complete the following:

1. Download the [MVTec 3D-AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad/downloads) dataset and place its unzipped contents (it should be a folder titled "mvtec_3d_anomaly_detection") in the "datasets" folder.
2. Run train script (`py -3 -m train`)

If you are having issues or if it is running slowly, run the systemtest script.

Also note that I trained these models on a NVIDIA GeForce RTX 3050 Ti Laptop GPU, so if you make some minor alterations to the train code (number of epochs or fixed_size for example), you can likely train better models than those found here. The models linked below only went through 8 epochs each, and ideally you want 11+ for this kind of model.

## Test
Run the test/visualize script with `py -3 -m test` or `py -3 -m test filenamehere.png`

### Example Results

#### Carrot

![a cut carrot](https://github.com/vmc-7645/anomaly-detection-3d/blob/main/testimg.png)

![anomaly pointcloud of the cut carrot](https://github.com/vmc-7645/anomaly-detection-3d/blob/main/pointcloudscreenshot_testimg.png)

#### Bagel

![a damaged bagel](https://github.com/vmc-7645/anomaly-detection-3d/blob/main/testimg1.png)

![anomaly pointcloud of the damaged bagel](https://github.com/vmc-7645/anomaly-detection-3d/blob/main/pointcloudscreenshot_testimg1.png)

## Models

Models uploaded to my drive, here: [anomaly-detection-3d-models](https://drive.google.com/drive/folders/1lfxbOMJv7Q0RX6g0ZDTO4yKyFxPAdj3O?usp=drive_link).

![train loss curve](https://github.com/vmc-7645/anomaly-detection-3d/blob/main/losscurve.png)
