# COVID19: Brazilian's pandemic peak forecasting 
This project contains an Exploratory Data Analysis of the evolution of 
the pandemic around the world. It shows the evolution of the pandemic for several countries
and their relation with the population.

Furthermore, it shows a forecasting model to predict the Brazilian's pandemic peak. Preliminary,
results estimate the pandemic peak for the 11th of June 2020.

The purposes of this project are two. From one side, I would like to share with the community my personal 
forecasting of the Brazilian's pandemic peak. And from the other side, I would like to share the structure
of this project for next more accurate forecasting and visualization.
 
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
The parameters are located in the yaml file (exp: params_1.yml). To run the model given the 
parameters params_1.yml you should run:

    python main.py params_1.yml 
Furthermore, it available a web app developed with streamlit (https://www.streamlit.io/).
The web app shows a simple visualization of my results. To shows the web app you should run:

    streamlit run run.py
The app folder contains:
- main.py: main program to train and test the model;
- params_1.yml: parameters used to train and test the model;
- learning_covid: package with functions related to preprocessing and modelling. 
Currently are available the RandomForestRegression, Ridge regression, Lasso regression,
 and kernel Ridge regression (several not-linear functions);
- models: joblib file with the Pipeline models;
- tables: several table saved during the training and the test;
- report: final results;
- run.py: web app with my personal results.




