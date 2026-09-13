import pandas as pd

def load_house_price_series(IN: str) -> pd.Series:
    df = pd.read_csv(f'{IN}/Average-prices-2026-03.csv', index_col=0, parse_dates=[0])
    df.drop(columns=['Area_Code', 'Monthly_Change', 'Annual_Change', 'Average_Price_SA'], inplace=True)
    df_eng = df[df['Region_Name'] == 'England']
    df_eng = df_eng.loc['1975-01-01':]
    # Taking sums of months for quarters
    df_eng = df_eng.resample('QE').mean(numeric_only=True)
    df_eng['Region_Name'] = 'England'
    # Move to front
    df_eng = df_eng[['Region_Name'] + [col for col in df_eng.columns if col != 'Region_Name']]
    # Converting to periods
    df_eng.index = df_eng.index.to_period('Q')
    # Keeping only columns we need
    df_eng = df_eng["Average_Price"]

    return df_eng