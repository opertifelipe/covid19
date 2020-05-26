# COVID19: Brazilian's pandemic peak forecasting 
This project contains an Exploratory Data Analysis of the evolution of 
the pandemic around the world. It shows the evolution of the pandemic for several countries
and their relation with the population.

Furthermore, it shows a forecasting model to predict the Brazilian's pandemic peak. Preliminary,
results estimate the pandemic peak for the 11th of June 2020.

## Structure of the project
The projects contains three folders:
- data: where are located all the data used in the project;
- notebook: jupyter notebook with the EDA analysis;
- app: where are located all the functions to train the models and forecast the pandemic 
peak. Furthermore, this folder contains a web app developed with *streamlit* that allows 
a simple visualization of the results.

### Data
All the data are extracted in from Kaggle with origin the 2019 Novel Coronavirus Visual Dashboard operated by the Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE). 
Link: https://www.kaggle.com/c/covid19-global-forecasting-week-5/data?select=train.csv

### Notebook
Notebook contains the EDA analysis of the evolution of the pandemic around the world. 
Furthermore, it trains a KernelRidge regression to predict the Brazilian's pandemic peak.

### App
The app folder contains the programs to train and test the model given a list of parameteres.
The parameters are located in the yaml file (exp: params_1.yml). 


 

