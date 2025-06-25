# Model


### What is the loss function used for training?
We use a custom loss function, which is a weighted sum of a binary cross-entropy and a mean squared error loss function. Each loss function is applied to a separate output channel of the neural network, so in that sense our methane detection model is a multi-task model.

The binary cross-entropy is obtained by thresholding the labels – pixels with a fractional reduction y*ᵢ* in the B12/B11 ratio that exceed the `binary_threshold` hyperparameter (set to 0.001) are encoded as 1's, and those below as 0's.

$$
b_i = \begin{cases} 1 & \text{if }|y_i| > T_{binary} \\ 0 & \text{if } |y_i| \le T_{binary} \end{cases}
$$

$$
\mathcal{L}_{BCE} = -  \sum_{i=1}^{N} \left[ b_i \log(\hat{p}_i) + (1 - b_i) \log(1 - \hat{p}_i) \right]
$$

The mean squared error loss is calculated only on pixels that exceed the `binary_threshold`hyperparameter, and it is weighted by the `MSE_multiplier` hyperparameter.

$$
\mathcal{L}_{MSE} = \sum_{i=1}^{N} b_i (y_i - \hat{\mu}_i)^2
$$

$$
\mathcal{L} = \mathcal{L}_{BCE} +W_{MSE}  \cdot \mathcal{L}_{MSE}
$$

### How should the model output be interpreted?

The U-net outputs two channels that together give a probabilistic prediction of methane.

The first is a segmentation channel which gives a binary prediction of the presence or absence of methane in each pixel. Presence is defined as the fractional reduction in the B12/B11 band ratio exceeding the `binary_threshold` value (0.001). Pass this channel through a sigmoid (inverse logistic) function to obtain a probability between 0 and 1\. This is the model's internal predictive probability of methane, but it should be used with caution, as it is calibrated on the training data, which contains many more methane plumes than would be encountered "in the wild". Changing the chip size (from 128x128 to 256x256) or the number or sizes of plumes inserted in each chip results in different output probabilities.

The second is a regression channel which gives a conditional prediction of the quantity of methane present in each pixel. In other words, it answers the question "if there is methane in this pixel, then how much?" The metric is the fractional change in the B12/B11 band ratio (frac), and still needs to be post-processed to obtain a gas concentration in mol/m². The loss function on the conditional channel only applies to pixels that contain methane, so outside of these areas, the model is free to conjure up possible methane plumes "just in case." The conditional channel on its own therefore tends to be noisy.

A "marginal" prediction can be obtained by multiplying the binary probability and the conditional prediction. This can be used like the output of a pixelwise regression model, as a point prediction of the quantity of methane in each pixel. But again, it is calibrated to the training data, and so should be interpreted cautiously.

A probabilistic interpretation that combines the two predictive channels is that the prediction for each pixel is modelled as a [zero-inflated](https://en.wikipedia.org/wiki/Zero-inflated_model) Gaussian distribution. The loss function is the likelihood of this distribution, and the optimizer's task is to maximize this likelihood. In this perspective, the `MSE_multiplier` parameter controls the σ parameter of the predictive Gaussian distribution. It is inversely proportional to the variance. Since frac values are small, informally this explains why `MSE_multiplier` needs to be high for the two loss components to be in tune with each other. The marginal prediction is the expected value of the zero-inflated distribution.

### Why isn't the model just a semantic segmentation? Why isn't the model just a pixelwise regression?

A semantic segmentation model (pixelwise binary classification) is insufficient to be able to report methane emissions from assets, as it only indicates the presence of a plume, but not its concentration. In order to quantify the emissions, one would have to use the segmentation output as a mask for a separately retrieved methane concentration map. This can be a successful strategy, for example Rouet-Leduc and Hulbert (2024) train a pixel classifier which they use to mask the Varon et al. (2021) "classical" retrievals, and thus reduce false positives.

We did not follow this strategy, as we found the classical retrievals to be noisy, sometimes even reporting negative methane concentrations on pixels classified as methane. We chose to leverage the power of neural networks for high-quality retrievals as well as masking.

A pixelwise regression model (where the neural network predicts the quantity of methane or frac in each pixel) could also be trained for this problem. The main issue here is that the ground truth is very sparse (most pixels don't contain methane), and so with a mean squared error loss most predictions are strongly regularized towards zero. Sometimes during model training the weights would even collapse to always predicting zero everywhere. We could only counter-balance this somewhat by giving higher weight to methane pixels, but this introduced an additional hyperparameter that is difficult to tune.

We therefore chose the multi-task approach (two-part loss) which gives the best of both worlds – separating mask and retrieval outputs with a neat probabilistic interpretation, and adapting well to the sparseness of the ground truth data.

### Is class imbalance a problem?

Though we often worried about class imbalance (most pixels do not contain methane), we found that the binary cross-entropy loss handled it well, and the trained models did not simply predict no methane everywhere. Experiments with using focal loss – which is designed to cope better with class imbalance –  yielded much worse results.

### Does the model ever infer negative methane concentrations?

Yes, this can happen, as the conditional prediction is unconstrained. Some (but not all) trained models predict negative methane in areas where it is quite sure there is no methane. Even though this is physically impossible, there is no penalty in the loss function for outputting a negative prediction on methane-free pixels.

If desired, it would be quite simple to constrain the model to only predict positive methane, by passing the conditional prediction layer through a sigmoid function before applying the masked mean squared error function. We have not tried this, as we found the unconstrained conditional predictions to generally behave well: we do not see significant negative concentrations in the marginal predictions.

### What model architecture are you using?

We are using an U-Net++ (Zhou et al. 2018\) with an efficientnet-b4 (Tan, Le, 2019\) encoder starting with “noisy-student” pretrained weights. Using the [Segmentation Models](https://github.com/qubvel-org/segmentation_models.pytorch) Python package this model can be initialized with

```python
import segmentation_models_pytorch as smp

model = smp.UnetPlusPlus(
    encoder_name="timm-efficientnet-b4",
    encoder_weights="noisy-student",
    in_channels=...,
    classes=...
)
```

### Have you tried any other model architectures?

We saw a large improvement going from an U-Net (Ronneberger et al. 2015\) with a ResNet-50 (He et al. 2016\) encoder to an U-Net++ (Zhou et al. 2018\) with an efficientnet-b4 (Tan, Le, 2019). Larger encoders, i.e. going from efficientnet-b1 to b2 to b3 to b4 gave us small, but significant improvements.

### How do you train the model?

The training loop (code [here](https://github.com/Orbio-Earth/Orbio/blob/main/methane-cv/src/training/training_script.py)) is fairly standard with some special design decisions.

We typically trained with 4 GPUs using distributed training on Azure ML. We use the optimizer AdamW with no weight decay and a batch size schedule (Smith et al. 2017), where the batch size is gradually increased over the first 30 warmup epochs from 16 to 160\.

Unlike most computer vision applications, because we can generate arbitrarily large synthetic datasets, using augmentations to juice as much information as possible out of every sample is not essential. So we only use simple non-destructive augmentations: random rotations by multiples of 90°, and random flips.

In earlier model training, we also implemented and used a signal modulation transformation that weakens the methane concentration by a random constant factor in each chip. We found that presenting the model with stronger plumes in early epochs, and gradually reducing them in later epochs, helped guide the optimizer. In more recent training runs, we noticed that the modulation was no longer needed to get good results.

Validation metrics are calculated on three pre-selected areas (see [Why have you selected these three specific regions to validate in?](Validation.md#why-have-you-selected-these-three-specific-regions-to-validate-in)) and training is only stopped if the validation metrics are not improving for multiple epochs (Early Stopping with patience of X).

As the main validation metric for Early Stopping, we selected average Recall over multiple emission buckets and averaged over the three regions. Recall is calculated using a classification threshold that allows for 10 false positive pixels on average over all 128x128 validation crops. A model checkpoint is saved if the mean recall improves. Balancing False Positives and Recall allows us to select the model that offers the highest sensitivity (i.e. best detection accuracy) without exceeding our target noise tolerance. See [here](https://github.com/Orbio-Earth/methane-cv/blob/main/src/training/training_script.py#L1378) for the implementation of validation and [here](Validation.md#what-metrics-do-you-use-during-model-training) for more details and validation metric visuals..

### How long does it take to train the model?

Around 15 hours depending on some randomness in the Early Stopping criteria. We have been using the Azure VM *Standard\_NC64as\_T4\_v3* (64 cores, 440 GB RAM, 2816 GB disk) with [4 x NVIDIA Tesla T4](https://www.nvidia.com/en-us/data-center/tesla-t4/) GPUs. Our typical throughput is in the 550-600 chips/second (a chip is a single data item) range during training, and over 2,000 chips/second during validation (where gradients do not need to be computed). By modern standards, these are not particularly powerful GPUs, but we find that because of the large number of bands, more powerful GPUs would not be used efficiently and the training quickly becomes IO-bound.

### Have you tried using foundation models?

Yes, we ran some experiments with pretrained weights from popular foundation models trained on generic computer vision tasks for Sentinel 2 data. We did not see any measurable improvement in model performance from those. Our informal interpretation is that the methane signal in bands 11 and 12 is very subtle, and its detection quite orthogonal to more typical computer vision tasks on satellite imagery, such as object detection, change detection and land cover classification.

### What experiments did not work?

We have found the performance to be not affected by the exact number of synthetic plumes we are inserting into training chips. Our default was to insert a random number of between 0-5 plumes into each chip, but we have seen similar results using 0-2, 0-3 and 0-4. Only 0-1 showed significantly worse results as now we seem to show too little plumes.

Also, at our current dataset size of around 1.5 million 128x128 chips, we have found the dataset size to not have an impact anymore on performance. There are large gains from going from 0.1 million to 0.75 million, but doubling the size again to 1.5 million yields no significant gains.

We tried using larger chips, going from 128x128 to 256x256 following the intuition that having more context and less border pixels could help the model in learning better. To our surprise, we got worse results. Also, using the same 256x256 chips, we tried using random crops of 224x224, 192x192 and 160x160 as another data augmentation technique. This did not work as well, the best results were achieved giving the full size chips.

### What are some of the main limitations of these models?

Despite significant strides in model performance over the last year, the primary limitation of the Sentinel 2 computer vision model remains false positives. Large methane plumes detectable by Sentinel 2 are relatively rare events, and so even a small rate of false positives can easily overwhelm the true detections. For this reason, we find that we continue to need careful quality assurance (QA) of candidate plumes in production.

In recent models trained on simulated plumes, we observed that the model would sometimes create false positives with very realistic-looking plume morphologies. Our interpretation is that a model that leverages morphological information also risks being prone to hallucinating more realistic plumes. These are therefore difficult to invalidate in QA. For this reason, we do not use these models in our production setting.

Just like the classical retrieval algorithms, our computer vision model relies heavily on reference scenes to infer the presence of methane on a target date. This works well when a scene is stable (such as in deserts), but breaks down if a scene changes rapidly (vegetation, agriculture, or large changes in soil moisture). In the classical algorithms, such changes cause false positives, whereas with computer vision they tend to raise the detection threshold. Another subtle consequence is that persistent sources can sometimes be erased if methane is present in one of the reference scenes, especially if the wind direction does not change between overpasses. We observed this in single blind release studies, where we found we needed to manually select alternative reference scenes to recover certain plumes.

### What are the most important next steps?

#### Reference chip selection

Our current [method](Data.md#how-do-you-select-reference-scenes) to select two reference scenes only uses a cloud masking threshold and proximity in time. When we examine scenes with known methane emissions, we find that plumes that are missed (false negatives) by the computer vision inference can be revealed with a manual selection of alternative reference scenes. This can happen if the source also emitted methane during one of the reference overpasses, which erases the methane signal on the target date. It can also happen if scenes further in the past are better references than more recent ones, for example in terms of soil moisture conditions. We therefore expect this to be fertile ground for model improvements, including ideas such as:

- feature engineering reference scenes, like taking averages of the last 5/10 overpasses;
- selecting the reference scenes by some similarity metric to the target date;
- allowing the model to choose its own reference scenes by some attention mechanism.

#### Hyperparameter tuning for simulated plumes

There are many decisions that need to be made when it comes to synthetic data generation with simulated plumes, including multiple hyperparameters that we have observed to be important to model performance, but have not yet systematically tuned. These are:

- the distributions of emission sizes in the training data,
- the diffusion and turbulence parameters of the [Gaussian plume models](Data.md#how-do-you-simulate-methane-plumes)
- the distribution of the number of plumes to include in each chip (we currently sample uniformly from 0 to 5 for each chip)

The development cycle for these parameters is long, as it includes plume simulations, synthetic data generation, model training and validation. But there may well significant detection performance improvements available if pursued.

## References

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 770-778).

Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." In *Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18*, pp. 234-241. Springer international publishing, 2015\.

Rouet-Leduc, Bertrand, and Claudia Hulbert. "Automatic detection of methane emissions in multispectral satellite imagery using a vision transformer." *Nature Communications* 15, no. 1 (2024): 3801\.

Smith, S. L., Kindermans, P. J., Ying, C., & Le, Q. V. (2017). Don't decay the learning rate, increase the batch size. *arXiv preprint arXiv:1711.00489*.

Tan, M., & Le, Q. (2019, May). Efficientnet: Rethinking model scaling for convolutional neural networks. In *International conference on machine learning* (pp. 6105-6114). PMLR.

Varon, Daniel J., Dylan Jervis, Jason McKeever, Ian Spence, David Gains, and Daniel J. Jacob. "High-frequency monitoring of anomalous methane point sources with multispectral Sentinel-2 satellite observations." *Atmospheric Measurement Techniques* 14, no. 4 (2021).

Zhou, Z., Rahman Siddiquee, M. M., Tajbakhsh, N., & Liang, J. (2018). Unet++: A nested u-net architecture for medical image segmentation. In *Deep learning in medical image analysis and multimodal learning for clinical decision support: 4th international workshop, DLMIA 2018, and 8th international workshop, ML-CDS 2018, held in conjunction with MICCAI 2018, Granada, Spain, September 20, 2018, proceedings 4 (pp. 3-11)*. Springer International Publishing.

#