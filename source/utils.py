import pandas as pd
from scipy.stats import kstest


def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df = df[:16758]
    df.replace("..", pd.NA, inplace=True)

    print("Missing value counts for each column:")
    # Missing value counts
    print(df.isnull().sum())

    columns_to_drop = [
        "Year", "Year Code", "Country Name", "Country Code",  
        "Diabetes prevalence (% of population ages 20 to 79) [SH.STA.DIAB.ZS]",
        "Antiretroviral therapy coverage for PMTCT (% of pregnant women living with HIV) [SH.HIV.PMTC.ZS]",
        "Capital health expenditure (% of GDP) [SH.XPD.KHEX.GD.ZS]",
        "Births attended by skilled health staff (% of total) [SH.STA.BRTC.ZS]",
        "Adults (ages 15+) and children (ages 0-14) newly infected with HIV [SH.HIV.INCD.TL]",
        "Literacy rate, adult total (% of people ages 15 and above) [SE.ADT.LITR.ZS]"        
    ]
    
    print("\nDropping columns with a lot of missing values.")
    # Drop unnecessary columns, or the ones with a lot of missing values
    df = df.drop(columns=columns_to_drop)
    # Drop the rows with missing values
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    # Data values are strings, convert to numeric types
    df = df.apply(pd.to_numeric, errors='coerce')
    # Clean column names
    df.columns = [
        'Adolescent fertility rate',
        'Adults and children living with HIV',
        'Antiretroviral therapy coverage',
        'Birth rate, crude',
        'Hospital beds',
        'Life expectancy at birth',
        'People using safely managed drinking water services',
        'Population growth',
        'Unemployment'
    ]

    return df


def kolmogorov_smirnov(df):
    # Perform Kolmogorov-Smirnov test for normality
    ks_results = {col: kstest(df[col].dropna(), 'norm', args=(df[col].mean(), df[col].std())) for col in df.columns}
    # Determine normality based on p-value threshold of 0.05
    print("Kolmogorov-Smirnov Test Results:\n")
    for col, result in ks_results.items():
        is_normal = 'Normal' if result[1] > 0.05 else 'Not Normal'
        print(f"{col}: {is_normal}")
