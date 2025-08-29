#%%
import numpy as np
import pandas as pd
from scipy.stats import expon
import scipy.stats as stats
from causal_curve import TMLE_Regressor
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from plotnine import ggplot, aes, geom_line, geom_ribbon, ggtitle, labs, ylim, xlim
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

#%%
# DataFrames for negative control
data = {
    'exposure': ['current', 'mov_avg2'],
    'coheficient': np.zeros(2),  
    'p-value': np.zeros(2)      
}

nc_rain = pd.DataFrame(data)
nc_temp = pd.DataFrame(data)

#%%
# Load data
df = pd.read_csv("D:/dataset.csv", encoding='latin-1')
df = df.iloc[:, 5:27]

#%%
# 1. Label Encoding DANE
le = LabelEncoder()
df['DANE_labeled'] = le.fit_transform(df['DANE'])
scaler = MinMaxScaler()
df['DANE_normalized'] = scaler.fit_transform(df[['DANE_labeled']])

# 2. Label Encoding DANE_year
le_year = LabelEncoder()
df['DANE_year_labeled'] = le_year.fit_transform(df['DANE_Year'])
scaler_year = MinMaxScaler()
df['DANE_year_normalized'] = scaler_year.fit_transform(df[['DANE_year_labeled']])

# 3. Label Encoding DANE_year_month
le_period = LabelEncoder()
df['DANE_period_labeled'] = le_period.fit_transform(df['DANE_period'])
scaler_period = MinMaxScaler()
df['DANE_period_normalized'] = scaler_period.fit_transform(df[['DANE_period_labeled']])

#%%
# Transform year and month
df.Year = df.Year - 2007
df["sin_month"] = np.sin(2 * np.pi * df["Month"] / 12)
df["cos_month"] = np.cos(2 * np.pi * df["Month"] / 12)

#%%
# Rain and temperature in t+1
df['rain_t1'] = df.groupby('DANE')['rain'].shift(-1)
df['temp_t1'] = df.groupby('DANE')['temp'].shift(-1)

#%%
# Moving average variables       
variables = ['rain', 'temp', 'SST12', 'SST3', 'SST34', 'SST4', 'NATL', 'SATL', 'TROP', 'soilmoi']
windows = [2]

for var in variables:
    for window in windows:
        # Create new column name
        nueva_col = f'{var}_mov_avg{window}'
        
        # Calculate moving average
        df[nueva_col] = df.groupby('DANE')[var].transform(
            lambda x: x.rolling(window=window, min_periods=1, closed='right').mean()
        )

#%%
# Confounders as binary transformation function
def transform_binary(df, columns_to_transform):
    for col in columns_to_transform:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = (df[col] > median_val).astype(int)
        else:
            print(f"Column {col} not found in dataframe")
    return df

#%%
# Define exposure types to process
exposure_types = ['current', 'mov_avg2']

# Process each exposure type
for exposure_type in exposure_types:
    
    # Define variable names based on exposure type
    if exposure_type == 'current':
        rain_var = 'rain'
        temp_var = 'temp'
        rain_cols = ['DANE_normalized', 'DANE_year_normalized', 'DANE_period_normalized', 'Year', 'sin_month', 'cos_month', 'SST12', 'SST3', 'SST34', 'SST4', 'NATL', 'SATL', 'TROP', 'forest', 'ENSO', 'rain', 'sir', 'rain_t1']
        temp_cols = ['DANE_normalized', 'DANE_year_normalized', 'DANE_period_normalized', 'Year', 'sin_month', 'cos_month', 'SST12', 'SST3', 'SST34', 'SST4', 'NATL', 'SATL', 'TROP', 'rain', 'soilmoi', 'forest', 'ENSO', 'temp', 'sir', 'temp_t1']
        rain_transform_cols = slice(6, 14)
        temp_transform_cols = slice(6, 16)
    else:  # mov_avg2
        rain_var = 'rain_mov_avg2'
        temp_var = 'temp_mov_avg2'
        rain_cols = ['DANE_normalized', 'DANE_year_normalized', 'DANE_period_normalized', 'Year', 'sin_month', 'cos_month', 'SST12_mov_avg2', 'SST3_mov_avg2', 'SST34_mov_avg2', 'SST4_mov_avg2', 'NATL_mov_avg2', 'SATL_mov_avg2', 'TROP_mov_avg2', 'forest', 'ENSO', rain_var, 'sir', 'rain_t1']
        temp_cols = ['DANE_normalized', 'DANE_year_normalized', 'DANE_period_normalized', 'Year', 'sin_month', 'cos_month', 'SST12_mov_avg2', 'SST3_mov_avg2', 'SST34_mov_avg2', 'SST4_mov_avg2', 'NATL_mov_avg2', 'SATL_mov_avg2', 'TROP_mov_avg2', 'rain_mov_avg2', 'soilmoi_mov_avg2', 'forest', 'ENSO', temp_var, 'sir', 'temp_t1']
        rain_transform_cols = slice(6, 14)
        temp_transform_cols = slice(6, 16)
    
    # Create dataframes for rainfall and temperature
    df_rain = df.loc[:, rain_cols]
    df_temp = df.loc[:, temp_cols]
    
    # Transform confounders to binary
    df_rain = transform_binary(df_rain, columns_to_transform=df_rain.columns[rain_transform_cols])
    df_temp = transform_binary(df_temp, columns_to_transform=df_temp.columns[temp_transform_cols])
    
    # Filter data based on exposure ranges
    # Rainfall between 0.1 and 60 mm
    df_rain_filtered = df_rain[df_rain[rain_var].between(0.01, 60)]
    df_rain_clean = df_rain_filtered.dropna()
    
    # Temperature between 15-30°C
    df_temp_filtered = df_temp[df_temp[temp_var].between(15, 30)]
    df_temp_clean = df_temp_filtered.dropna()
    
    # Determine index for updating results
    idx = 0 if exposure_type == 'current' else 1
    
    # Process Rainfall
    print(f"Processing rainfall - {exposure_type}")
    
    # TMLE for rainfall
    tmle_rain = TMLE_Regressor(n_estimators=1500, random_seed=123, bandwidth=999, verbose=True, 
                               max_depth=5, learning_rate=0.0001)
    
    # Prepare data
    X_cols_rain = df_rain_clean.columns[:15]  # First 15 columns as covariates
    X_data_rain = df_rain_clean[X_cols_rain]
    T_data_rain = df_rain_clean[rain_var]
    y_data_rain = df_rain_clean['sir']
    
    # Fit model
    tmle_rain.fit(T=T_data_rain, X=X_data_rain, y=y_data_rain)
    
    # Calculate confidence intervals
    tmle_results_rain = tmle_rain.calculate_CDRC(0.95)
    
    # Plot exposure-response curve
    plot_rain = (
        ggplot(aes(x=tmle_results_rain.Treatment, y=tmle_results_rain.Causal_Dose_Response)) 
        + geom_line() 
        + geom_ribbon(aes(ymin=tmle_results_rain.Lower_CI, ymax=tmle_results_rain.Upper_CI), alpha=0.1, fill='blue')
        + labs(x='Rainfall mm', y='SIR') 
        + ggtitle(f'{exposure_type}')
    )
    print(plot_rain)
    
    # Negative control test
    interpolate_func = interp1d(
        tmle_results_rain['Treatment'], 
        tmle_results_rain['Causal_Dose_Response'], 
        kind='linear', 
        fill_value="extrapolate"
    )
    
    df_rain_clean_copy = df_rain_clean.copy()
    df_rain_clean_copy['predicted_sir'] = interpolate_func(df_rain_clean_copy[rain_var])
    
    # Gamma regression for negative control
    y = df_rain_clean_copy['sir']
    X = df_rain_clean_copy[['predicted_sir', 'rain_t1']]
    X = sm.add_constant(X)
    
    gamma_model_rain = sm.GLM(y, X, family=sm.families.Gamma())
    gamma_result_rain = gamma_model_rain.fit()
    
    # Extract coefficient and p-value for rain_t1
    coefficient_rain_t1 = gamma_result_rain.params[2]
    pvalue_rain_t1 = gamma_result_rain.pvalues[2]
    
    # Update negative control dataframe
    nc_rain.loc[idx, 'coheficient'] = coefficient_rain_t1
    nc_rain.loc[idx, 'p-value'] = pvalue_rain_t1
    
    # Calculate effect size measures
    effect_at_001 = tmle_rain.point_estimate(np.array([0.01]))
    effect_at_60 = tmle_rain.point_estimate(np.array([60]))
    effect_diff_rain = effect_at_60 - effect_at_001
    std_sir_rain = df_rain_clean['sir'].std()
    cohens_d_rain = effect_diff_rain / std_sir_rain
    
    lower_ci_rain = tmle_results_rain['Lower_CI'].iloc[0]
    upper_ci_rain = tmle_results_rain['Upper_CI'].iloc[0]
    se_rain = (upper_ci_rain - lower_ci_rain) / (2 * 1.96)
    
    # Store results for EValue analysis
    if exposure_type == 'current':
        effect_diff_rain_current = effect_diff_rain
        cohens_d_rain_current = cohens_d_rain
        lower_ci_rain_current = lower_ci_rain
        upper_ci_rain_current = upper_ci_rain
        se_rain_current = se_rain
    else:  # mov_avg2
        effect_diff_rain_mov_avg2 = effect_diff_rain
        cohens_d_rain_mov_avg2 = cohens_d_rain
        lower_ci_rain_mov_avg2 = lower_ci_rain
        upper_ci_rain_mov_avg2 = upper_ci_rain
        se_rain_mov_avg2 = se_rain
    
    # Process Temperature
    print(f"Processing temperature - {exposure_type}")
    
    # TMLE for temperature
    tmle_temp = TMLE_Regressor(n_estimators=1500, random_seed=123, bandwidth=999, verbose=True, 
                               max_depth=5, learning_rate=0.0001)
    
    # Prepare data
    X_cols_temp = df_temp_clean.columns[:17]  # First 17 columns as covariates
    X_data_temp = df_temp_clean[X_cols_temp]
    T_data_temp = df_temp_clean[temp_var]
    y_data_temp = df_temp_clean['sir']
    
    # Fit model
    tmle_temp.fit(T=T_data_temp, X=X_data_temp, y=y_data_temp)
    
    # Calculate confidence intervals
    tmle_results_temp = tmle_temp.calculate_CDRC(0.95)
    
    # Plot exposure-response curve
    plot_temp = (
        ggplot(aes(x=tmle_results_temp.Treatment, y=tmle_results_temp.Causal_Dose_Response)) 
        + geom_line() 
        + geom_ribbon(aes(ymin=tmle_results_temp.Lower_CI, ymax=tmle_results_temp.Upper_CI), alpha=0.1, fill='blue')
        + labs(x='Temperature °C', y='SIR') 
        + ggtitle(f'{exposure_type}')
    )
    print(plot_temp)
    
    # Negative control test
    interpolate_func = interp1d(
        tmle_results_temp['Treatment'], 
        tmle_results_temp['Causal_Dose_Response'], 
        kind='linear', 
        fill_value="extrapolate"
    )
    
    df_temp_clean_copy = df_temp_clean.copy()
    df_temp_clean_copy['predicted_sir'] = interpolate_func(df_temp_clean_copy[temp_var])
    
    # Gamma regression for negative control
    y = df_temp_clean_copy['sir']
    X = df_temp_clean_copy[['predicted_sir', 'temp_t1']]
    X = sm.add_constant(X)
    
    gamma_model_temp = sm.GLM(y, X, family=sm.families.Gamma())
    gamma_result_temp = gamma_model_temp.fit()
    
    # Extract coefficient and p-value for temp_t1
    coefficient_temp_t1 = gamma_result_temp.params[2]
    pvalue_temp_t1 = gamma_result_temp.pvalues[2]
    
    # Update negative control dataframe
    nc_temp.loc[idx, 'coheficient'] = coefficient_temp_t1
    nc_temp.loc[idx, 'p-value'] = pvalue_temp_t1
    
    # Calculate effect size measures
    effect_at_15 = tmle_temp.point_estimate(np.array([15]))
    effect_at_30 = tmle_temp.point_estimate(np.array([30]))
    effect_diff_temp = effect_at_30 - effect_at_15
    std_sir_temp = df_temp_clean['sir'].std()
    cohens_d_temp = effect_diff_temp / std_sir_temp
    
    lower_ci_temp = tmle_results_temp['Lower_CI'].iloc[0]
    upper_ci_temp = tmle_results_temp['Upper_CI'].iloc[0]
    se_temp = (upper_ci_temp - lower_ci_temp) / (2 * 1.96)
    
    # Store results for EValue analysis
    if exposure_type == 'current':
        effect_diff_temp_current = effect_diff_temp
        cohens_d_temp_current = cohens_d_temp
        lower_ci_temp_current = lower_ci_temp
        upper_ci_temp_current = upper_ci_temp
        se_temp_current = se_temp
    else:  # mov_avg2
        effect_diff_temp_mov_avg2 = effect_diff_temp
        cohens_d_temp_mov_avg2 = cohens_d_temp
        lower_ci_temp_mov_avg2 = lower_ci_temp
        upper_ci_temp_mov_avg2 = upper_ci_temp
        se_temp_mov_avg2 = se_temp

# Save negative control results
nc_rain.to_csv("D:/nc_rain.csv", index=False)
nc_temp.to_csv("D:/nc_temp.csv", index=False)

# Create EValue parameter dataframes for rainfall
analyses_rain = {
    'current': {
        'Effect': effect_diff_rain_current,
        'Cohen_s_d': cohens_d_rain_current,
        'Lower_CI': lower_ci_rain_current,
        'Upper_CI': upper_ci_rain_current,
        'SE': se_rain_current
    },
    'mov_av2': {
        'Effect': effect_diff_rain_mov_avg2,
        'Cohen_s_d': cohens_d_rain_mov_avg2,
        'Lower_CI': lower_ci_rain_mov_avg2,
        'Upper_CI': upper_ci_rain_mov_avg2,
        'SE': se_rain_mov_avg2
    },
}

param_evalue_rain = pd.DataFrame(analyses_rain).T.reset_index()
param_evalue_rain.columns = ['Analysis', 'Effect', 'Cohen_s_d', 'Lower_CI', 'Upper_CI', 'SE']
param_evalue_rain = param_evalue_rain[['Analysis', 'Effect', 'Cohen_s_d', 'Lower_CI', 'Upper_CI', 'SE']]
param_evalue_rain['Effect'] = param_evalue_rain['Effect'].astype(float)
param_evalue_rain['Cohen_s_d'] = param_evalue_rain['Cohen_s_d'].astype(float)
param_evalue_rain.to_csv("D:/param_evalue_rain.csv", index=False)

# Create EValue parameter dataframes for temperature
analyses_temp = {
    'current': {
        'Effect': effect_diff_temp_current,
        'Cohen_s_d': cohens_d_temp_current,
        'Lower_CI': lower_ci_temp_current,
        'Upper_CI': upper_ci_temp_current,
        'SE': se_temp_current
    },
    'mov_av2': {
        'Effect': effect_diff_temp_mov_avg2,
        'Cohen_s_d': cohens_d_temp_mov_avg2,
        'Lower_CI': lower_ci_temp_mov_avg2,
        'Upper_CI': upper_ci_temp_mov_avg2,
        'SE': se_temp_mov_avg2
    },
}

param_evalue_temp = pd.DataFrame(analyses_temp).T.reset_index()
param_evalue_temp.columns = ['Analysis', 'Effect', 'Cohen_s_d', 'Lower_CI', 'Upper_CI', 'SE']
param_evalue_temp = param_evalue_temp[['Analysis', 'Effect', 'Cohen_s_d', 'Lower_CI', 'Upper_CI', 'SE']]
param_evalue_temp['Effect'] = param_evalue_temp['Effect'].astype(float)
param_evalue_temp['Cohen_s_d'] = param_evalue_temp['Cohen_s_d'].astype(float)
param_evalue_temp.to_csv("D:/param_evalue_temp.csv", index=False)