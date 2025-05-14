# Adversarial Curation

This is the implementation of the experiments presented in our paper:

**Self-Consuming Generative Models with Adversarially Curated Data**

 *Paper coming soon on arXiv*  
<!-- Replace this line with a link when available: [arXiv:xxxx.xxxxx](https://arxiv.org/abs/xxxx.xxxxx) -->

---

## Project Structure
```
.
├── cirfar10_malicious.py         # Adversarial preference attack for CIFAR-10
├── cirfar10_reward.py            # Reward model for CIFAR-10
├── retrain_ddpm.py               # Main retraining script for DDPM
├── malicious.py                  # Adversarial preference attack for toy data
├── reward.py                     # Reward model for toy data
├── ddpm-torch/                   # Customized ddpm
├── data/, models/                # Data, pretrained models
```

## Setup

```bash
git clone https://github.com/Wei-0125/Adversarial-Curation.git
cd Adversarial-Curation
pip install -r requirements.txt
```

Alternatively, using a Conda environment is supported. 

## Example Commands
#### 1. **Fully synthetic curated retraining without malicious attack on Gaussian-8 dataset**

```
python retrain_ddpm.py --name Project --dataset_name gaussian8 --fully_synthetic --filter \
    --network ./models/ddpm/gaussian8/ddpm_gaussian8.pt
```

#### 2. **Fully synthetic curated retraining with malicious attack (gradient) on CIFAR-10 dataset**

```
python retrain_ddpm.py --name Project --dataset_name cifar10 --fully_synthetic --filter --malicious \
    --network ./models/ddpm/cifar10/ddpm_cifar10.pt \
    --buget 0.2 --alpha 0.8
```

#### 3. **Mixed curated retraining with malicious attack (gradient) on Gaussian-8 dataset**

```
python retrain_ddpm.py --name Project --dataset_name gaussian8 --filter --malicious \
    --network ./models/ddpm/gaussian8/ddpm_gaussian8.pt \
    --original_dataset ./models/ddpm/gaussian8/samples.npy \
    --prop_gen_data 0.8
```

#### 4. **Fully synthetic curated retraining with malicious attack (random) on CIFAR-10 dataset**

```
python retrain_ddpm.py --name Project --dataset_name cifar10 --fully_synthetic --filter --malicious --random \
    --network ./models/ddpm/cifar10/ddpm_cifar10.pt \
    --buget 0.2
```

## Related Codebases
This project builds upon and extends several excellent open-source codebases:

- [tqch/ddpm-torch](https://github.com/tqch/ddpm-torch): baseline ddpm implementation.
- [chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models): pretrained CIFAR-10 classifiers used in reward model.
- [QB3/gen_models_dont_go_mad](https://github.com/QB3/gen_models_dont_go_mad): implementation reference for adversarial retraining loops.
