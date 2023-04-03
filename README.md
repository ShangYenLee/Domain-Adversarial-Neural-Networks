# Domain-Adversarial-Neural-Networks
DANN has two main objectives, the first one is to accurately predict the classification label of the input image in the target domain. The second one is to reduce the difference between the source and target domain.

||MNIST-M → SVHN|MNIST-M → USPS|
|:---:|:---:|:---:|
|Trained on source|0.33940|0.73522|
|Adaptation (DANN)|0.41745|0.82392|
|Trained on target|0.93466|0.98992|
<p align="center">
<img src="https://drive.google.com/uc?id=1tZco1w0pC69kDw_BH7Ly8eLxk1829BQu" width="40%" hspace="12"/>
<img src="https://drive.google.com/uc?id=1pgJR1iWSiW3fUoT0cSW5ZqaenljJdM3W" width="40%" hspace="12"/>
</p>

<p align="center">
<img src="https://drive.google.com/uc?id=1THwEu5FVWykGaFXmPQ3vQWVJet-cyknN" width="40%" hspace="12"/>
<img src="https://drive.google.com/uc?id=1UFJxITkvxitkN0bqn6DXjGRgvohjmyKf" width="40%" hspace="12"/>
</p>

<p align="center">
(a) latent space by class &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(b) latent space by domain
</p>
