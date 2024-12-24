# CPM_EEG_Craving

This repository provides all the scripts and code required to perform **Connectome-Based Predictive Modeling (CPM)** analysis. CPM is a powerful method for predicting individual behavior using brain connectivity data.  
This repository supports the paper **"An Electroencephalography Connectome Predictive Model of Craving for Methamphetamine."**

## Background

The CPM method is a widely used approach for leveraging brain connectivity data to predict behavior. Details about the CPM method can be found in the original publication:  
[Shen et al., Nature Protocols (2016)](https://www.nature.com/articles/nprot.2016.178.pdf).

The methodology and this repository are inspired by and adapted from the work of:  
1. **Xilin Shen**  
   - *Using connectome-based predictive modeling to predict individual behavior from brain connectivity*  
   - Nature Protocols, 12(3), 506-518, 2017.  
   - Citation: Shen X., Finn E.S., Scheinost D., Rosenberg M.D., Chun M.M., Papademetris X., Constable R.T.

2. **Yumeng Ju**  
   - GitHub repository: [https://github.com/jym115](https://github.com/jym115)

## Repository Contents

This repository includes:
- **CPM Analysis Scripts:** Scripts for processing input data and running CPM analysis.  
- **Performance Evaluation Functions:** Tools to assess model performance and accuracy.  
- **Visualization Tools:** Scripts for generating visual representations of results.  
- **Mediation Analysis Scripts:** Additional analysis scripts for examining mediation effects.

## Preparation

### Input Data Requirements
To perform CPM analysis, you need two key datasets:

1. **Predictor Matrix (MxN):**  
   - `M`: Number of features (e.g., brain connectivity metrics).  
   - `N`: Number of subjects.  
   - This matrix contains features (e.g., connectivity measures) used as predictors across all subjects.

2. **Outcome Vector (Nx1):**  
   - `N`: Number of subjects.  
   - This vector includes the predicted outcome values (e.g., craving scores) for each subject.

#### For EEG Functional Connectivity Data
In neuroscience applications, particularly with EEG data, the **Predictor Matrix** may have a `MxMxN` structure:
- `M x M`: Connectivity metrics between nodes, ROIs, or sensors.  
- `N`: Number of participants.

### Example Data Preparation
For EEG data:
1. Compute the connectivity metrics between nodes (e.g., correlation, coherence) for each subject.
2. Format the data as a `MxMxN` matrix.
3. Prepare an outcome vector (`Nx1`) with corresponding behavioral or craving scores.

### Getting Started
1. Ensure your data matches the required format.
2. Use the scripts in this repository to preprocess your data.
3. Follow the CPM pipeline to train and evaluate your model.

## Author and Acknowledgements

This repository was created by **Hangbin Zhang** and is inspired by the work of:  
- **Xilin Shen** ([Shen et al., 2017](https://www.nature.com/articles/nprot.2016.178.pdf))  
- **Yumeng Ju** ([GitHub Repository](https://github.com/jym115/CPM_MDD_long_term))

## Citation

If you use this repository, please cite:  
Shen, X., Finn, E.S., Scheinost, D., Rosenberg, M.D., Chun, M.M., Papademetris, X., & Constable, R.T. (2017).  
*Using connectome-based predictive modeling to predict individual behavior from brain connectivity.*  
Nature Protocols, 12(3), 506.
