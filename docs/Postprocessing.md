# Postprocessing


### How are individual plumes masked (segmented) out of the inference output?

The masking stage in the methane detection pipeline is designed to convert sensor outputs into binary masks that isolate methane plume signals. Our masking is performed on a binary probability map that represents the likelihood of methane presence at each pixel. We use a watershed segmentation algorithm, a classical image processing technique that treats the probability map as a topographic surface. High-probability pixels are treated as peaks, and the algorithm simulates a flood from selected seed points to delineate coherent plume structures. Parameters such as minimum distance between markers, thresholding for plume regions, and gradient-based seed selection are tuned to optimize segmentation across different sensors and models. The final output is a binary raster mask, with pixels labeled as 1 indicating methane-positive detections and zero representing methane-negative pixels.

We experimented with doing the segmentation on marginal and conditional retrievals as well, but found that the binary probability map produced the best plume delineation. The implementation of the masking can be found in the methane-cv repo here:  https://github.com/Orbio-Earth/Eucalyptus-code-archive/blob/main/methane-cv/sbr_2025/utils/prediction.py#L225

### How do you estimate the emissions rate for a plume?

Plume quantification is performed using the **Integrated Mass Enhancement (IME)** method, which is the standard approach for estimating methane emission rates from satellite observations, see Varon et al. (2018). IME represents the total methane enhancement within a segmented plume and is combined with wind data to estimate the source emission rate (**Q**). The core relationship is defined as:

**Q \= IME / τ**

where **τ** is the effective residence time of methane within the detectable plume. This residence time is operationally defined as the plume length divided by wind speed, or:

**τ \= L / U\_eff**

Here, **L** is the effective plume length, calculated as the longest side of the bounding rectangle that encloses the segmented plume area, and **U\_eff** is the effective wind speed, obtained from ERA5 or GEOSS meteorological reanalysis data at the time and location of the plume.

The final formula used to estimate methane emissions, expressed in kilograms per hour, is:

**Q \= (U\_eff / L) × IME × 57.75286**

This equation incorporates the molecular weight of methane (16.04246 g/mol) and converts the emission rate from moles per second to kilograms per hour using the constant 57.75286.

The **IME** itself is calculated by summing the methane enhancements (in mol/m²) across all pixels identified as plume-positive in the binary mask and multiplying by the pixel area (derived from the spatial resolution of the raster). Only pixels with enhancement values above zero are included. The raster used for this summation is the **rescaled retrieval**, which contains column methane values in physical units (the rescaled retrieval is explained in the section below).

The quantification module in the methane-cv repo is located here: https://github.com/Orbio-Earth/Eucalyptus-code-archive/blob/main/methane-cv/src/inference

### Why do you use the rescaled retrieval for quantification?

Classical methods for methane detection and emissions quantification directly produce a "retrieval" raster of methane concentration point estimates for each pixel in a scene. This same retrieval is used for masking and then quantification of the masked retrieval.

With the computer vision approach, the output is probabilistic, with interpretation further discussed in the [Data](Data.md) section. The prediction is in two parts: the binary probability and the conditional frac, which is converted into a conditional retrieval by inverting the radiative transfer laws.

When it comes to masking, we have already discussed that we perform the masking on the binary probability channel, rather than the retrieval. But what is the analog of the masked retrieval used for quantification? There are two obvious options, but they both have significant downsides.

**Masking the conditional retrieval.** Once a plume has been accepted, via a QA or automated process, the conditional retrieval answers the question "if there is methane in this pixel, then how much?" so it therefore seems natural to mask the conditional retrieval for quantification. But experiments with controlled releases showed that resulting quantifications significantly overestimate the ground truth emission rate, with a mean absolute percentage error measured above \+50% in 8 controlled releases. This is because the tail of the plume, where the presence of methane above the threshold has low probability, gets treated as if it definitely contained methane. Visually, these masked plumes with boosted tails also have cliff edges at their perimeter, which is suggestive of the problem.

**Masking the marginal retrieval.** The marginal retrieval is the conditional retrieval weighted by the binary probability, and is an analog to the output of the classical retrieval algorithms. By definition, it is lower than the conditional retrieval, and so results in lower quantifications. Visually, it also has smoother tails, without the cliff edge effect observed in the masked conditional retrieval. However, for a plume with low likelihood, the quantification is in effect weighted down by the model's probability of the plume not existing at all. One can think of this as a form of regularization. If such a plume is nevertheless selected in QA, this downweighting is inappropriate, and leads to underestimating the emission rate if the plume is real. It is also troubling that the binary probability is internally calibrated to the computer vision model's training data (more frequent plumes in training data \= generally higher predicted probabilities), with no direct real-world interpretation, and this bias will translate directly to the quantification.

The downsides of the conditional and marginal retrievals for quantification motivated the development of the **rescaled retrieval**, which is obtained *after masking*, by dividing the marginal retrieval by the maximum binary probability *within each plume*. This has several advantages:

1. the tails of the plumes are not boosted inappropriately;
2. there is no cliff edge at the perimeter of the plume; and
3. the quantification is not highly sensitive to the frequency of plumes in the training data.

There is an (admittedly fuzzy) probabilistic interpretation of the rescaled retrieval. The binary prediction for each pixel can be factored into two components: (1) the probability of the methane plume that this pixel belongs to being real, and (2) the conditional probability of this particular pixel exceeding the threshold given that the plume is real. By dividing by the maximum pixel probability within the plume, we are eliminating the first factor, because we are quantifying a plume assuming it is real. But the second factor remains, and tempers down the estimated concentrations in the more uncertain tails of the plume.

### What sources of wind data can be used?

Any wind data source can be used, as long as it provides reliable wind speed and direction estimates near the surface, ideally with spatial resolution down to at least around 25km and temporal resolution of 1–3 hours. The better the resolution, the more accurate the emission quantification.

We typically use [GEOS-FP (NASA GMAO)](https://gmao.gsfc.nasa.gov/GMAO_products/NRT_products.php) and cross-check against [ERA5 (ECMWF Reanalysis)](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5). Both are freely available, provide global coverage, and offer spatial resolution of around 25km. GEOS-FP has a 3-hour temporal resolution, while ERA5 provides data hourly. While ERA5 is generally considered more accurate, in practice we’ve found both datasets yield similar results for methane detection. The bigger opportunity for improvement would come from using higher-resolution wind data to better capture local wind behavior. That said, in a real-time production setting, ERA5 is often less practical due to slower availability and longer download times.

#