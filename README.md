# MTSCCleav: a Multivariate Time Series Classification (MTSC)-based Method for Predicting Human Dicer Cleavage Sites

<img src="figures/hsa-let-7a-1_ss.svg.pptx.svg" alt="Predicted secondary structure of the sequence S of pri-miRNA hsa-let-7a-1." width="400">

This figure shows the predicted secondary structure of the sequence of pre-miRNA "hsa-let-7a-1". 
- The colors on the nodes reflect the probability of the base pair in this predicted secondary structure.
- The two scissor symbols indicate the two Dicer cleavage sites on 5' arm (i.e., the 56-57 bond) and 3' arm (i.e., the 27-28 bond).
- We would like to predict whether a subsequence contains a cleavage site at the center.

<img src="figures/pipeline.pptx.svg" alt="The overall pipeline of this study." width="400">

The overall pipeline of this study.
MTSCCleav consists of three parts: time series encoding, time series transformation (encoding), and classification. 
- Nine encoding methods to convert RNA data to time series were introduced.
- Five ROCKET-based methods were used for time series transformation. 
- Ridge Classifier was used for classification.

<!-- ## Notifications -->

## Installation

```
conda create -n mtsccleav python=3.12
conda activate mtsccleav
pip install sktime
pip install --upgrade numba
pip install matplotlib
pip install seaborn
pip install biopython
# https://viennarna.readthedocs.io/en/latest/api_python.html
python -m pip install viennarna
pip install -U aeon[all_extras]
# If "no matches found" error appears, try below command.
pip install -U aeon"[all_extras]"
# For testing lightgbm classifier
```

Other useful commands
```
# Delete the environment if needed.
conda env remove -n mtsccleav
# List the existing environments
conda env list
# Deactivate the environment
conda deactivate
# Export your active environment to a new file:
conda env export > environment.yml
# Readers can make new environment from our environment. 
# But different platforms may have their own platform-specific packages that may cause error if importing `environments.yml` directly.
conda env create --name envname --file=environments.yml
```

## Project Structure

- Readers should visit the files in this order: [prepare_datasets.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/prepare_datasets.ipynb) → [classify.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/classify.ipynb) → [plot_cd.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/plot_cd.ipynb) → [interpret.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/interpret.ipynb).
- To get familiar with the time series libraries, go to [classify_aeon.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/classify_aeon.ipynb) and [classify_sktime.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/classify_sktime.ipynb).
- To explore about ROCKET-based classifiers, go to [rocket_convolution_examples.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/rocket_convolution_examples.ipynb) and [hydra_convolution.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/hydra_convolution.ipynb).
```bash
.
├── code
│   ├── classify_aeon.ipynb # Testing convolution-based classifiers using aeon library
│   ├── classify_sktime.ipynb # Testing convolution-based classifiers using sktime library
│   ├── classify.ipynb
│   ├── hydra_convolution.ipynb # Convolution used in Hydra
│   ├── interpret.ipynb # Check which subsequence in the multivariate time series is important for classification
│   ├── mtsccleav.py # General library used for this project
│   ├── plot_cd.ipynb
│   ├── prepare_datasets.ipynb # Prepare the miRNA dataset
│   ├── rocket_convolution_examples.ipynb # Convolution examples for ROCKET
│   ├── sota # Compare the result with the SOTA, DiCleave
│   │   ├── DiCleave-data # Store our training data for DiCleave and results returned by DiCleave
│   │   ├── DiCleave-main # This is created by unzipping "DiCleave-main-e512d74.zip"
│   │   ├── DiCleave-main-e512d74.zip # With the latest commit (e512d74), accessed on 2025-06-17
│   │   ├── evaluate_dicleave.ipynb
│   │   └── run_and_measure.sh # It should be placed inside folder "DiCleave-main" for correct path.
│   ├── test_transformations.ipynb
│   └── transformations.py # time series transformation methods
├── data
├── environment.yml # Conda environment file.
├── figures
└── README.md
```
## Corresponding Paper

[Coleman Yu, Raymond Chi-Wing Wong, and Tatsuya Akutsu,  
"MTSCCleav: a Multivariate Time Series Classification (MTSC)-based method for predicting human Dicer cleavage sites",  
IEEE Access, pp11048 - 11063, Vol.14, 2026](https://ieeexplore.ieee.org/document/11359221)  
([pdf](https://github.com/colemanyu/time-series-classification-cleavage-paper/blob/main/time_series_classification_cleavage_paper.pdf))([code](https://github.com/colemanyu/time-series-classification-cleavage))(slides)([TeX](https://github.com/colemanyu/time-series-classification-cleavage-paper))

### Data/Code for the Figures/Tables

Figures
- Figure 4: [rocket_convolution_examples.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/rocket_convolution_examples.ipynb)
- Figure 5: [hydra_convolution.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/hydra_convolution.ipynb)
- Figures 6, 7, 8: [plot_cd.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/plot_cd.ipynb)
- Figure 9: [interpret.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/interpret.ipynb) 

Tables
- Tables 1, 2, 3: [prepare_datasets.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/prepare_datasets.ipynb)
- Table 4: [transformations.py](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/transformations.py), [test_transformations.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/test_transformations.ipynb)
- Tables 6, 7: [classify.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/classify.ipynb)
- Table 8: [classify.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/classify.ipynb), [evaluate_dicleave.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/sota/evaluate_dicleave.ipynb)

## Resources

Other studies related to human Dicer cleavage site prediction (Newest first):
- [DiCleavePlus](https://github.com/MGuard0303/DiCleavePlus) (Different Problem Setting)
- [DiCleave](https://github.com/MGuard0303/DiCleave) (SOTA)
- [ReCGBM](https://github.com/ryuu90/ReCGBM) (Previous work of SOTA)

Some useful time series libraries:
- [sktime](https://www.sktime.net/en/stable/)
  - [Introduction to sktime](https://www.sktime.net/en/latest/examples/00_sktime_intro.html) 
  - [Time Series Classification, Regression, Clustering & More](https://www.sktime.net/en/stable/examples/02_classification.html)
  - [Demo of ROCKET transform](https://www.sktime.net/en/latest/examples/transformation/rocket.html)
- [tslearn](https://tslearn.readthedocs.io/en/stable/)
- [aeon](https://www.aeon-toolkit.org/en/stable/index.html)
  - [Convolution based time series classification in aeon](https://www.aeon-toolkit.org/en/stable/examples/classification/convolution_based.html)
  - [The ROCKET transform](https://www.aeon-toolkit.org/en/latest/examples/transformations/rocket.html)
  - [MiniRocket](https://www.aeon-toolkit.org/en/stable/examples/transformations/minirocket.html)
- [tsai](https://timeseriesai.github.io/tsai/)
  - [ROCKET: a new state-of-the-art time series classifier](https://colab.research.google.com/github/timeseriesAI/tsai/blob/master/tutorial_nbs/02_ROCKET_a_new_SOTA_classifier.ipynb) | [Tutorial notebooks](https://timeseriesai.github.io/tsai/tutorials.html)
- [pyts](https://pyts.readthedocs.io/en/stable/index.html)
- [DTAIDistance](https://dtaidistance.readthedocs.io/en/latest/) (Distance Measure)
- [stumpy](https://stumpy.readthedocs.io/en/latest/) (Matrix Profile)
- [SAX-VSM](https://jmotif.github.io/sax-vsm_site/) (a.k.a. jmotif, in Java)
    
Some biological computation libraries:
- [ViennaRNA](https://viennarna.readthedocs.io/en/latest/index.html)
- [Biopython](https://biopython.org/)

## MISC

For testing tsai:
```
pip install tsai
# Downgrade fastcore for compatibility with tsai 
pip install fastcore==1.5.29
# Downgrade scikit for compatibility with tsai 
pip install scikit-learn==1.1.3
conda create -n tsai_env python=3.10 -y
conda activate tsai_env
pip install tsai==0.3.6 fastai==2.7.12 fastcore==1.5.29 scikit-learn==1.1.3
pip install --force-reinstall numpy pandas scikit-learn fastai
pip install --force-reinstall fastcore==1.5.29
pip install xgboost
pip install lightgbm
```