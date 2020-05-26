import streamlit as st
from learning_covid.preprocessing.preprocessing import create_dataframe_country_target
import pandas as pd
import altair as alt

import numpy as np

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


st.title("COVID19: Brazilian's pandemic peak forecasting")

#@st.cache
def load_data(data_file):
    data = pd.read_csv(data_file)
    return data
data =  load_data("../data/train.csv")
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
data = data[(data["County"].isnull()) & (data["Province_State"].isnull())]

df_countries = create_dataframe_country_target(data, "Brazil", "Fatalities",countries_dict).reset_index()
for country in list(countries_dict.keys()):
    if country!="Brazil":
        df_country = create_dataframe_country_target(data, country, "Fatalities",countries_dict).reset_index()
        df_countries = df_countries.append(df_country)
df_countries = df_countries.set_index("Country_Region")

"""
## Plot deaths for each country
"""

countries = st.multiselect("Choose countries", list(df_countries.index), ["Italy","Brazil","US"])
data_deaths = df_countries.loc[countries]
data_deaths["Country"] = list(data_deaths.index)
#st.write(data_deaths[data_deaths["Country"] == "US"].head())

countries_death = (
    alt.Chart(data_deaths)
    .mark_line(opacity=0.5)
        .encode(
        x="Date:T",
        y="Rate_over_population:Q",
        color="Country:N"
    )
)
st.altair_chart(countries_death, use_container_width=True)

data_model = pd.read_csv("tables/prediction_several_days_roll_model_kernel_ridge_target_Fatalities_n_days_30_target_deaths_Rate_over_population.csv")
data_comparison = data_model.copy()

data_target = data_model[["Date","Target_real"]]
data_target.columns = ["Date","Deaths"]
data_target["Status"] = "Real"
data_predicted = data_model[["Date","Target_predicted"]]
data_predicted.columns = ["Date","Deaths"]
data_predicted["Status"] = "Predicted"

data_model = data_target.append(data_predicted)
data_model.index = data_model["Status"]


status = st.multiselect("Choose between real and predicted", list(data_model.index), ["Real","Predicted"])
data_forecast = data_model.loc[status]



prediction = (
    alt.Chart(data_model)
    .mark_line()
        .encode(
        x="Date:T",
        y="Deaths:Q",
        color="Status:N"
    )
)
st.altair_chart(prediction, use_container_width=True)

peak_day = data_comparison.iloc[data_comparison["Target_predicted"].argmax()]["Date"]
"""
Peak max in Brazil will be the: **"""+peak_day+"""**
"""
