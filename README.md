# Project Eucalyptus

This repository contains 
- Trained models to detect methane plumes for Sentinel‑2, Landsat 8/9 and EMIT.
- End‑to‑end notebooks covering inference and post‑processing to get from raw imagery to position and emission rates of methane plumes.
- Synthetic‑plume toolkit that injects physically realistic methane signatures into clean scenes for training and stress‑testing.


# Who is Orbio?
[Orbio Earth](https://www.orbio.earth/) is a climate-tech company founded in 2021 to turn open satellite data into actionable methane intelligence.

Project Eucalyptus is authored by the Orbio Earth team—Maxime Rischard, Timothy Davis, Zani Fooy, Mahrukh Niazi, Philip Popien, Robert Edwards,, Vikram Singh, Jack Angela, Thomas Edwards, Maria Navarette, Diehl Sillers, Wojciech Adamczyk and Robert Huppertz — with support from external collaborators.

# What can I find where?
- [Detection and Quantification on Sentinel-2](notebooks/sentinel2.ipynb)
- [Detection and Quantification on Landsat 8/9](notebooks/landsat.ipynb)
- [Detection and Quantification on EMIT](notebooks/emit.ipynb)
- [Synthetically inserting methane to create training data](notebooks/synthetic_training_data_generation_demo.ipynb)


# Running the notebooks

The notebooks require python version 3.10.
All necessary dependencies can be installed from `requirements.txt`.
For example, to install these into a virtual environment:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notebooks can be accessed through jupyter lab, included as a dependency, or through a software of your choice.

```
jupyter lab
```

# How to contribute or ask questions 

Please use Github Issues if you want to contribute or ask questions. 
We are open to contributions for which we request you to either create a PR or discuss it within a Github Issue.