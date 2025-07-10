# Gated Time Series Forecasting with Channel Interactions

Main contributions in this repo include :
- Implementation of the TSMixer model done in [1] with experiments being run according to the best parameters configured in the paper.
- Implementation of a new time series forecasting model based on a gating mechanism.
- Implementation of Patched TSMixer feeding input as patches to TSMixer based on [1] & [8]

## Baselines to compare with
- General Code structure, trasoformers experiments and Linear models experiments are taken from https://github.com/cure-lab/LTSF-Linear which is implementation for the paper "Are Transformers Efficient for Time Series Forecasting?(AAAI 2023) [2]
- results are compared to transformers [3,4,5,6,7], Linear models [2], TSMixers [1] and the newly proposed PatchTST[8].

## Detailed Description
We provide all experiment script files in `./scripts`:
| Files      |                              Interpretation                          |
| ------------- | -------------------------------------------------------| 
| EXP-LongForecasting      | Long-term Time Series Forecasting Task                    |
| EXP-LookBackWindow      | Study the impact of different look-back window sizes   | 

Other github repos which the code is based on :

The implementation of Autoformer, Informer, Transformer is from https://github.com/thuml/Autoformer

The implementation of FEDformer is from https://github.com/MAZiqing/FEDformer

The implementation of Pyraformer is from https://github.com/alipay/Pyraformer

## Gating-based Time Series Forecasting Model
![Alt text](pics/Gated%20Time%20Series%20Forecasting%20Model.png)


## Patches-based TSMixer Forecasting Model
![Alt text](pics/PatchTSMixer%20Model.png)

## Getting Started
### Environment Requirements

First, please make sure you have installed Conda. Then, our environment can be installed by:
```
conda create -n LTSF_Linear python=3.6.9
conda activate LTSF_Linear
pip install -r requirements.txt
```

### Data Preparation

You can obtain all the four benchmarks from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided in Autoformer. All the datasets are well pre-processed and can be used easily.

```
mkdir dataset
```
**Please put them in the `./dataset` directory**

### Quick overview of structure
- In `scripts/ `, different experiments for different models can be executed.
- In `models/`, All previous implementations alongside the three newly implemented models are provided.
- In `logs/`, Results for different models are stored in different folders under this directory.


# References

[1] Tolstikhin, Ilya O., et al. "Mlp-mixer: An all-mlp architecture for vision." Advances in neural information processing systems 34 (2021): 24261-24272.
[2] Zeng, Ailing, et al. "Are transformers effective for time series forecasting?." arXiv preprint arXiv:2205.13504 (2022).
[3] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).
[4] Wu, Haixu, et al. "Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting." Advances in Neural Information Processing Systems 34 (2021): 22419-22430.
[5] Zhou, Haoyi, et al. "Informer: Beyond efficient transformer for long sequence time-series forecasting." Proceedings of the AAAI conference on artificial intelligence. Vol. 35. No. 12. 2021.
[6] Zhou, Tian, et al. "Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting." International Conference on Machine Learning. PMLR, 2022.
[7] Liu, Shizhan, et al. "Pyraformer: Low-complexity pyramidal attention for long-range time series modeling and forecasting." International conference on learning representations. 2021.
[8] Nie, Yuqi, et al. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." arXiv preprint arXiv:2211.14730 (2022).