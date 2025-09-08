# MTSCCleave: a Multivariate Time Series Classification (MTSC)-based Method for Predicting Human Dicer Cleavage Sites
<!-- https://stackoverflow.com/questions/39777166/display-pdf-image-in-markdown -->
<!-- for d in *.pdf ; do inkscape --without-gui --file=$d --export-plain-svg=${d%.*}.svg ; done -->
<!-- ![Predicted secondary structure of the sequence S of pri-miRNA “hsa-let-7a-1".](figures/hsa-let-7a-1_ss.svg.pptx.svg) -->
<img src="figures/hsa-let-7a-1_ss.svg.pptx.svg" alt="Predicted secondary structure of the sequence S of pri-miRNA hsa-let-7a-1." width="400" height="300">

This figure shows the predicted secondary structure of the sequence of pre-miRNA "hsa-let-7a-1". The two scissor symbols indicates the two cleavage sites on 5' arm and 3' arm.
- The colors on the nodes reflect the probability of the base pair in this predicted secondary structure.
- The 27-28 bond (i.e., the bond between 27th nucleotide and 28th nucleotide) and the 56-57 bond are the cleavage sites.
- The "scissors" are the human Dicer.

<!-- ![The overall pipeline of this study.](figures/pipeline.pptx.svg) -->
<img src="figures/pipeline.pptx.svg" alt="The overall pipeline of this study." width="400" height="300">

The overall pipeline of this study.
MTSCCleav consists of three parts: time series encoding, time series transformation, and classification. 
- We introduced nine encoding methods to convert RNA data to time series.
- Five ROCKET-based methods were used for time series transformation. 
- Ridge Classifier was used for classification.

# Notifications
Dates on [AoE](https://www.timeanddate.com/time/zones/aoe) Time Zone
- 2025-07-14 Submitted to [BMC Bioinformatics](https://bmcbioinformatics.biomedcentral.com/).

## Pending Tasks
- [] Upload the datasets to https://timeseriesclassification.com after publication.
- [] Make a video (YouTube) presentation.

# Install
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
# Project Structure
- Important folders and files in this repository are listed as belows:
- Readers should visit the files in this order: [prepare_datasets.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/prepare_datasets.ipynb) -> [classify.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/classify.ipynb) -> [plot_cd.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/plot_cd.ipynb) -> [interpret.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/interpret.ipynb).
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
└── README.md # This page.
```
# Corresponding paper
## Figures/ Tables in the paper
### Figures
- Figure 4: [rocket_convolution_examples.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/rocket_convolution_examples.ipynb)
- Figure 5: [hydra_convolution.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/hydra_convolution.ipynb)
- Figures 6, 7, 8: [plot_cd.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/plot_cd.ipynb)
- Figure 9: [interpret.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/interpret.ipynb) 

### Tables
- Tables 1, 2, 3: [prepare_datasets.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/prepare_datasets.ipynb)
- Table 4: [transformations.py](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/transformations.py), [test_transformations.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/test_transformations.ipynb)
- Tables 6, 7: [classify.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/classify.ipynb)
- Table 8: [classify.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/classify.ipynb), [evaluate_dicleave.ipynb](https://github.com/cyuab/time-series-classification-cleavage/blob/main/code/sota/evaluate_dicleave.ipynb)

# Resources
Other studies related to human Dicer cleavage site prediction (Newest first):
- [DiCleavePlus](https://github.com/MGuard0303/DiCleavePlus)
- [DiCleave](https://github.com/MGuard0303/DiCleave)
- [ReCGBM](https://github.com/ryuu90/ReCGBM)

Some useful time series libraries:
- [sktime](https://www.sktime.net/en/stable/)
- [tslearn](https://tslearn.readthedocs.io/en/stable/)
- [aeon](https://www.aeon-toolkit.org/en/stable/index.html)
- [tsai](https://timeseriesai.github.io/tsai/)
- [pyts](https://pyts.readthedocs.io/en/stable/index.html)
- [DTAIDistance](https://dtaidistance.readthedocs.io/en/latest/) (Distance Measure)
- [stumpy](https://stumpy.readthedocs.io/en/latest/) (Matrix Profile)
- [SAX-VSM](https://jmotif.github.io/sax-vsm_site/) (a.k.a. jmotif, in Java)
    
Some biological computation libraries:
- [ViennaRNA](https://viennarna.readthedocs.io/en/latest/index.html)
- [Biopython](https://biopython.org/)



# Contacts
- It will be updated after paper acceptance.

# MISC
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