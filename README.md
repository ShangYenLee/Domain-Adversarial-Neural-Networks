# Domain-Adversarial-Neural-Networks
DANN has two main objectives, the first one is to accurately predict the classification label of the input image in the target domain. The second one is to reduce the difference between the source and target domain.

## Training
* train DANN for target domain 'SVHN'
```
python train.py --data_root $data_dir --train_mode dann --target_file svhn
```
* train DANN for target domain 'USPS'
```
python train.py --data_root $data_dir --train_mode dann --target_file usps
```
* trained on sorce
```
python train.py --data_root $data_dir --train_mode sorce --target_file svhn/usps
```
* trained on target
```
python train.py --data_root $data_dir --train_mode target --target_file svhn/usps
```
## Download checkpoint
[checkpoint link](https://drive.google.com/drive/folders/1kUAP0tcsRnJz4vyVp5khdVVdd2N_yqIG?usp=sharing)
```
bash download.sh
```
## Inference DANN
```
python inference.py --checkpoint $checkpoint_dir --test_loc $test_folder --out_path $output_dir
```
```
bash inference.sh $test_folder $output_dir
```
## Result
<div align="center">

||MNIST-M → SVHN|MNIST-M → USPS|
|:---:|:---:|:---:|
|Trained on source|0.33940|0.73522|
|Adaptation (DANN)|0.41745|0.82392|
|Trained on target|0.93466|0.98992|
</div>
<p align="center">
<img src="https://drive.google.com/uc?id=1tZco1w0pC69kDw_BH7Ly8eLxk1829BQu" width="45%" hspace="12"/>
<img src="https://drive.google.com/uc?id=1pgJR1iWSiW3fUoT0cSW5ZqaenljJdM3W" width="45%" hspace="12"/>
</p>

<p align="center">
<img src="https://drive.google.com/uc?id=1THwEu5FVWykGaFXmPQ3vQWVJet-cyknN" width="45%" hspace="12"/>
<img src="https://drive.google.com/uc?id=1UFJxITkvxitkN0bqn6DXjGRgvohjmyKf" width="45%" hspace="12"/>
</p>

<p align="center">
(a) latent space by class &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(b) latent space by domain
</p>
