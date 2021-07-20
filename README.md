## tropicalcyclone_MLP
[![DOI](https://zenodo.org/badge/370519440.svg)](https://zenodo.org/badge/latestdoi/370519440)

Scientists have been searching for decades for breakthroughs in tropical cyclone intensity modeling to provide more accurate and timely tropical cyclone warnings. To this end, we developed a deep learning-based predictive model for North Atlantic 24-hour and 6-hour intensity forecast. We simulated 2019 and 2020 tropical cyclones as if in an operational forecast mode, and found that the model’s 24-hour intensity forecast outperformed some of the most skillful operational models by 5-22%. Also, the 6-hour intensity model (a lightweight version) produced realistic intensity labels for synthetic tropical cyclone tracks. These results highlight the potential for using deep neural network-based models to improve operational hurricane intensity forecasts and synthetic tropical cyclone generation.

MLP model performance illustration. More details in paper.

![Alt text](https://github.com/wenweixu/tropicalcyclone_MLP/blob/main/figs/Fig%202.png)

## how to cite:
W. Xu, K. Balaguru, A. August, N. Lalo, N. Hodas, M. DeMaria, & D. Judi. "Deep Learning Experiments for Tropical Cyclone Intensity Forecasts," Weather and Forecasting, (2021). DOI: 10.1175/WAF-D-20-0104.1


## training/validation/testing data
Supporting data should be downloaded from: 

Xu, Wenwei, Balaguru, Karthik, August, Andrew, Lalo, Nicholas, Hodas, Nathan, DeMaria, Mark, & Judi, David. (2021). Supporting data for Xu et al. 2021 - Weather and Forecasting (Version 1) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.4784610

The downloaded data can be stored under the folder `hurricane_data`. 

24-hour model data used in LOYO testing (before scaling):

`hurricane_data/NOAA_reanalysis_vars_global_w_dvs24.csv` 
and
`hurricane_data/NOAA_operational_vars_global_w_dvs24.csv`

24-hour model data used in LOYO testing (after scaling):

`hurricane_data/train_global_fill_REA_na_wo_img_scaled.csv`

2019 operational data reserved for 24-hour model independent test:

`hurricane_data/NOAA_operational_vars_global_wLabels_fill_na_Y2019.csv`

2020 operational data reserved for 24-hour model independent test:

`hurricane_data/NOAA_operational_vars_global_wLabels_fill_na_Y2020.csv`

6-hour model data used in LOYO testing (before scaling):

`hurricane_data/NOAA_reanalysis_vars_global_w_dvs6.csv`


## get started
Setup python environment from the `environment.yml` file:

```conda env create -f environment.yml```

Note that although a GPU version of tensorflow is used here, the model runs with a CPU version of tensorflow as well. Modify the `environment.yml` file if you don't have a GPU.

## model training and testing
To perform a LOYO test on the 24-hour forecast model

```python loyo_testing.py <architecture> <numpy seed>  <tensorflow seed>```

for example:

```python loyo_testing.py mlp 1 1```

To run the model with multiple combinations of numpy and tensorflow random seeds, a batch script similar to `randomseed.bat` can be used.

The testing results will be in `LOYO_results/seeds`. The results include two csv files: one csv file for testing statistics at annual level, and the other csv file for true label (y_test) with corresponding model predicted result (y_predict).


## funding acknowledgment
The operational forecast portion of this research was supported by the Deep Science Agile Initiative at Pacific Northwest National Laboratory (PNNL). It was conducted under the Laboratory Directed Research and Development Program at PNNL. PNNL is a multiprogram national laboratory operated by Battelle for the U.S. Department of Energy under contract DE-AC05-76RL01830.

The synthetic tropical cyclone portion of this research was supported by the Multisector Dynamics program areas of the U.S. Department of Energy, Office of Science, Office of Biological and Environmental Research as part of the multi-program, collaborative Integrated Coastal Modeling (ICoM) project.

K. B acknowledges support from the Regional and Global Modeling and Analysis Program of the U.S. Department of Energy, Office of Science, Office of Biological and Environmental Research (BER) and from NOAA's Climate Program Office, Climate Monitoring Program (Award NA17OAR4310155). 


## Contact
For details of the model development methodology, data source, and potential applications, please refer to our paper. Additional questions can be directed to Wenwei Xu (`wenwei.xu@pnnl.gov`).
