from learning_covid.preprocessing.preprocessing import create_training_countries
from learning_covid.preprocessing.preprocessing import create_training_df
from learning_covid.preprocessing.preprocessing import create_dataframe_country_target
from learning_covid.model.model import apply_model
from learning_covid.model.model import roll_predictions
import pandas as pd
from joblib import dump, load
from sklearn.metrics import mean_squared_error
from math import sqrt
import json


countries_dict = {"Italy": {
    "mean_age": 45.5,
    "GDP": 1988636},
    "Brazil": {
        "mean_age": 32.6,
        "GDP": 1847020},
    "US": {
        "mean_age": 38.1,
        "GDP": 21439453},
    "United Kingdom": {
        "mean_age": 40.5,
        "GDP": 2743586},
    "Portugal": {
        "mean_age": 42.2,
        "GDP": 236408},
    "Canada": {
        "mean_age": 42.2,
        "GDP": 1730914},
    "Austria": {
        "mean_age": 44.0,
        "GDP": 447718},
    "Belgium": {
        "mean_age": 41.4,
        "GDP": 517609},
    "Germany": {
        "mean_age": 47.1,
        "GDP": 3863344},
    "Greece": {
        "mean_age": 44.5,
        "GDP": 214012},
    "Finland": {
        "mean_age": 42.5,
        "GDP": 269654},
    "France": {
        "mean_age": 41.4,
        "GDP": 2707074},
    "Netherlands": {
        "mean_age": 42.6,
        "GDP": 902355},
    "Russia": {
        "mean_age": 39.6,
        "GDP": 1637892},
    "Poland": {
        "mean_age": 40.7,
        "GDP": 565854},
    "Sweden": {
        "mean_age": 41.2,
        "GDP": 528929},
    "Spain": {
        "mean_age": 42.7,
        "GDP": 1397870},
    "Japan": {
        "mean_age": 47.3,
        "GDP": 5154475},
    "Korea, South": {
        "mean_age": 41.8,
        "GDP": 1629532},
    "China": {
        "mean_age": 37.4,
        "GDP": 14140163

    }

}

def main():
    # Parameters
    target = "Fatalities"
    n_days = 30
    target_deaths = "Rate_over_population"
    model_to_apply = "kernel_ridge"
    country_to_test = "Brazil"
    day_to_roll = 100
    start_row = 60
    noise = 0.00

    # Read data
    data = pd.read_csv("../data/train.csv")

    # Remove data from counties and province. I am only interested in Countries data
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    data = data[(data["County"].isnull()) & (data["Province_State"].isnull())]

    # Prepare the training dataset using several countries. Brazil will be used as test.
    # I decided to use 30 days to predict the number of death at day 31
    countries_to_train = list(countries_dict.keys())
    countries_to_train.remove(country_to_test)
    df_train = create_training_countries(countries_to_train, data, target, target_deaths, n_days, countries_dict)
    df_train.to_csv("tables/df_train_target_"+target+"_n_days_"+str(n_days)+"_target_deaths_"+target_deaths+".csv",index=False)

    # Apply model
    CV_regr = apply_model(model_to_apply, df_train.drop(["Target", "Date"], axis=1), df_train["Target"])
    dump(CV_regr, "models/model_"+model_to_apply+"_target_"+target+"_n_days_"+str(n_days)+"_target_deaths_"+target_deaths+".joblib")

    # Prediction for one day
    df_test = create_training_df(create_dataframe_country_target(data, country_to_test, target, countries_dict), n_days,
                                        target=target_deaths)

    df_test["Prediction"] = CV_regr.predict(df_test.drop(["Target", "Date"], axis=1))
    df_test.to_csv("tables/prediction_one_day_model_"+model_to_apply+"_target_"+target+"_n_days_"+str(n_days)+"_target_deaths_"+target_deaths+".csv",index=False)

    # Prediction for several days (roll)
    df_test = create_training_df(create_dataframe_country_target(data, country_to_test, target, countries_dict), n_days,
                                 target=target_deaths)
    df_comparison = roll_predictions(day_to_roll, start_row, df_test, noise, CV_regr)
    df_comparison.to_csv("tables/prediction_several_days_roll_model_"+model_to_apply+"_target_"+target+"_n_days_"+str(n_days)+"_target_deaths_"+target_deaths+".csv",index=False)

    # Calculate the RMSE of the prediction
    true = df_comparison.dropna()['Target_predicted']
    pred = df_comparison.dropna()['Target_real']
    rmse = sqrt(mean_squared_error(true, pred))

    peak_day = df_comparison.iloc[df_comparison["Target_predicted"].argmax()]["Date"].strftime("%Y-%m-%d")

    # Create report
    report = {  "target" : "Fatalities",
                "n_days" : 30,
                "target_deaths" : "Rate_over_population",
                "model_to_apply" : "kernel_ridge",
                "country_to_test" : "Brazil",
                "day_to_roll" : 100,
                "start_row" : 60,
                "noise" : 0.00,
                "RMSE": rmse,
                "peak_day" : peak_day}
    with open("report/report_several_days_roll_model_"+model_to_apply+"_target_"+target+"_n_days_"+str(n_days)+"_target_deaths_"+target_deaths+".json", 'w') as fp:
        json.dump(data, fp)

if __name__ == "__main__":
    main()