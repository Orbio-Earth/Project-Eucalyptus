# Introduction


Methane monitoring has reached an inflection point. Multispectral and hyperspectral satellites now revisit every major production basin multiple times per week, delivering raw data with the spatial and spectral resolution to resolve individual methane emission plumes. Translating that stream of pixels into *quantified emission detections,* still demands a complex tool‑chain and deep domain knowledge across  satellite data process*ing,* advanced machine learning, radiative transfer physics, synthetic‑data generation and many more. The industry has got to a point where different providers of data will deliver different outputs (detections, quantifications, etc) when processing the exact same raw satellite imagery. The siloed nature of development in the industry is creating confusion and in some cases distrust amongst those who would benefit most from high quality emissions data.

**Project Eucalyptus is an attempt to close that gap.**
We are publishing a comprehensive tutorial to how Orbio has built its end-to-end methane pipelines over the last couple of years. Next to an extensive Q&A-like documentation, we are sharing large amounts of Orbio Earth’s production‑grade methane‑analytics pipeline—models, tooling, validation notebooks and reference datasets—released as open source under a non‑commercial licence. We are placing on the public square the same tools we use every day to alert producers when a thief hatch is left open or a flare goes out. No screenshots, no “request a demo”—actual weight files, notebooks and reproducible experiments.

### What is this doc?

A Q&A-like tutorial covering a wide range of lessons learned across training data, model training, tiling grids, cloud masks, validation of models, plume masking and quantification. In the [wider repo](https://github.com/Orbio-Earth/Project-Eucalyptus), and [code dump](https://github.com/Orbio-Earth/Eucalyptus-code-archive), you will find:

* **Trained models** for Sentinel‑2, Landsat 8/9 and EMIT, exportable to any modern framework.
* **End‑to‑end notebooks & scripts** covering inference, post‑processing, and objective detection‑threshold evaluation.
* **Synthetic‑plume toolkit** that injects physically realistic methane signatures into clean scenes for training and stress‑testing.

All components are released under a non‑commercial licence, see [here](https://github.com/Orbio-Earth/Project-Eucalyptus/blob/main/LICENSE). Commercial users: talk to us—there’s plenty we can build together. Contact: info@orbio.earth

### Why are we publishing this?

**Opening the Black Box.** Operators, investors and regulators increasingly rely on satellite methane estimates to steer day-to-day decisions. But as methane is a really hard problem and current best practice models are still known to cause issues (e.g. false positives or high error bars), users deserve to *inspect the math* behind the numbers.

**Methodological convergence.** Today, every team reinvents radiative transfer tricks, cloud masks and performance evaluation protocols. By publishing our approach to many of the complex problems in methane and satellites,  we want to support a dialog towards methodological convergence and help bring everybody closer to “the truth”.

**Accelerate innovation.** The synthetic‑plume engine and ready‑trained networks let newcomers skip months to years of plumbing and jump straight to *new* ideas—new models, better ways of benchmarking quantification accuracy, and casual discoveries. Progress compounds when the baseline is shared.

**Planet‑scale impact.** The climate clock is loud. Opening our stack is the fastest way to build trust, accelerate improvements to models and multiply our impact.

### Who is Orbio?

[Orbio Earth](https://orbio.earth) is a climate-tech company founded in 2021 to turn open satellite data into actionable methane intelligence.

Project Eucalyptus is authored by the Orbio Earth team—Maxime Rischard, Timothy Davis, Zani Fooy, Mahrukh Niazi, Philip Popien, Robert Edwards, Vikram Singh, Jack Angela, Thomas Edwards, Maria Navarette, Diehl Sillers, Wojciech Adamczyk and Robert Huppertz— with support from external collaborators.