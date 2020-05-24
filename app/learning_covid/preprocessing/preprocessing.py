import pandas as pd
import numpy as np

def create_dataframe_country_target(data, country, target, countries_dict):
    """
    This function create a dataframe for a country
    Args:
        - data: dataframe with all the data
        - country: country to analyze
        - targte: target to add. Could be "Fatalities" or "ConfirmedCases"
        - countries_dict: dictionary with additional info
    Return:
        A dataframe for the country with the target value e additional info

    """

    df_country = data[(data.Country_Region == country) & (data.Target == target)]
    df_country = df_country.groupby(["Country_Region", "Target", "Date"]).agg({"Population": "max",
                                                                               "TargetValue": "sum"
                                                                               })
    df_country = df_country.reset_index().set_index("Date", drop=True)
    df_country["mean_age"] = countries_dict[country]["mean_age"]
    df_country["GDP"] = countries_dict[country]["GDP"]
    df_country["Rate_over_population"] = df_country["TargetValue"] * 100000 / df_country["Population"]
    return df_country



def create_training_df(df_country,n_days,target,population=True,mean_age=True,gdp=True):
    """
    This function create a dataframe for training the models based in the number of lookback days.
    Args:
        - df_country: dataframe of the country
        - n_days: number of days as lookback
        - target: target column
        - population: if True population data will be used
        - mean_age: if True mean age data will be used
        - gdp: if True gdp data will be used
    Return:
        A dataframe that could be trained in a machine learning model
    """
    df_country = df_country.sort_index().reset_index(drop=False)
    columns = ["Target","Date"]+list(np.arange(0,n_days,1))
    df_train = pd.DataFrame(columns=columns)
    j=0
    for i in range(n_days,df_country.shape[0]):
        df_train = df_train.append({'Target': df_country.loc[i,target],
                                    'Date': df_country.loc[i,"Date"]
                                   }, ignore_index=True)
        for day in range(n_days):
            df_train.loc[j,day] = df_country.loc[i-n_days+day,target]
        j=j+1
    if population:
        df_train["Population"] = df_country["Population"].max()
    if mean_age:
        df_train["mean_age"] = df_country["mean_age"].max()
    if gdp:
        df_train["GDP"] = df_country["GDP"].max()
    return df_train

def create_training_countries(countries_to_train, data, target, target_deaths, n_days, countries_dict):
    """
    This function create the training dataset for several countries
    Args:
        - countries_to_train: list with countries to use
        - data: original data
        - target: target to use. Ex: "Fatalities"
        - target_deaths: target of deaths to use
        - n_days: number of days to lookback
        - countries_dict: additional information
    Return:
        A dataframe with the traing data using several countries
    """
    df_train = create_training_df(create_dataframe_country_target(data, countries_to_train[0], target,countries_dict),n_days, target=target_deaths)
    for i in range(1, len(countries_to_train)):
        df_train = pd.concat([df_train,
                         create_training_df(create_dataframe_country_target(data,
                                                                            countries_to_train[i],
                                                                            target,
                                                                            countries_dict),
                                            n_days,
                                            target=target_deaths)])
    return df_train