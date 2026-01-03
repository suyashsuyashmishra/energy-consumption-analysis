# importing modules
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend on the base module FIRST
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from flask import Flask, render_template

app = Flask(__name__)

# Helper function for Gini calculation
def gini_coefficient(x):
    n = len(x)
    x_sorted = np.sort(x)
    cumulative_x = np.cumsum(x_sorted)
    gini = ((n + 1) - 2 * np.sum(cumulative_x) / cumulative_x[-1]) / n
    return gini

@app.route('/')
def home():
    analyze_energy_trends()
    return render_template('index.html')

def analyze_energy_trends():
    # import data from csv file
    df = pd.read_csv("./energy.csv")
    print(df.head())
    print(df.describe())

    # check for duplicated rows
    duplicates = df.duplicated()
    if duplicates.any():
        print("There are duplicated rows in the dataset.")
    else:
        print("No duplicated rows found in the dataset.")
        
    # check for missing values
    missing_values = df.isnull().sum()
    print("Missing values in each column:\n", missing_values)
    df1 = df.drop(columns=['Unnamed: 0'])
    print(df1)

    # reducing the skewness of the entire dataset using log transformation
    df2 = df1.select_dtypes(include=[np.number]).apply(lambda x: np.log1p(x))
    print(df2.head())

    # Detect outliers using z score method
    from scipy import stats
    df3 = df2.drop(columns=['Year'])
    z_scores = stats.zscore(df3)
    abs_z_scores = np.abs(z_scores)
    outliers_z = df3[(abs_z_scores > 3).any(axis=1)]
    print("Outliers detected using Z-score method:\n", outliers_z)

    # Boxplot to visualize outliers
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df3, orient="h")
    plt.title('Boxplot to Visualize Outliers in Energy Data')

    # Noramalise the data with scaling factor like population or GDP
    df4 = df1.copy()
    df4['Energy_per_Capita'] = df4['Energy_consumption'] / df4['Population']
    df4['Energy_per_GDP'] = df4['Energy_consumption'] / df4['GDP']
    print(df4[['Country', 'Year', 'Energy_per_Capita', 'Energy_per_GDP']].head())

    # outlier analysis of the dataset
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df4[['Energy_per_Capita', 'Energy_per_GDP']], orient="h")
    plt.title('Boxplot to Visualize Outliers in Normalized Energy Data')
    plt.tight_layout()
    plt.savefig("./static/Boxplot to Visualize Outliers in Normalized Energy Data.png", dpi=300, bbox_inches='tight')

    # Trend analysis of energy consumption over the years
    # remove world value from country column for better visualization
    df_new = df4[df4['Country'] != 'World']
    plt.figure(figsize=(8, 4))
    # import only top 20 countries by energy consumption for clarity
    top_countries = df_new.groupby('Country')['Energy_consumption'].mean().sort_values(ascending=False).head(20).index
    sns.lineplot(data=df4, x='Year', y='Energy_consumption', hue='Country', hue_order=top_countries, ci=None)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Trend of Energy Consumption Over the Years by Country')
    plt.xlabel('Year')
    plt.ylabel('Energy Consumption')
    plt.tight_layout()
    plt.savefig("./static/Trend of Energy Consumption Over the Years by Country.png", dpi=300, bbox_inches='tight')

    # Correlation analysis
    plt.figure(figsize=(8, 6))
    correlation_matrix = df4.select_dtypes(include=[np.number]).corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix of Energy Data')
    plt.tight_layout()
    plt.savefig("./static/Correlation Matrix of Energy Data.png", dpi=300, bbox_inches='tight')

    # Rank countries by energy consumption
    ranked_energy = df_new.groupby('Country')['Energy_consumption'].mean().sort_values(ascending=False)
    print("Ranked countries by average energy consumption:\n", ranked_energy)
    
    # Visualize top 10 countries by energy consumption
    plt.figure(figsize=(10, 5))
    sns.barplot(x=ranked_energy.head(10).index, y=ranked_energy.head(10).values, ci=0.95)
    plt.title('Top 10 Countries by Average Energy Consumption')
    plt.xlabel('Country')
    plt.ylabel('Average Energy Consumption')
    plt.tight_layout()
    plt.savefig("./static/Top 10 Countries by Average Energy Consumption.png", dpi=300, bbox_inches='tight')

    # create model to predict energy consumption based on other features
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression 
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.metrics import mean_squared_error, r2_score

    # Features and target
    col = ["Energy_consumption", "Population", "GDP", "Energy_per_Capita", "Energy_per_GDP","Energy_production","CO2_emission"]
    df5 = df4[col].fillna(df4[col].median())
    missing_values_df5= df5.isnull().sum()
    print("Missing values in each column of df4:\n", missing_values_df5)
    df6 = np.log1p(df5[col])
    features = df6[['Population', 'GDP', 'Energy_production', 'CO2_emission']]
    target = df6['Energy_consumption']

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Predictions
    y_pred = model.predict(x_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test.values, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title('Actual vs Predicted Energy Consumption')
    plt.xlabel('Actual Energy Consumption')
    plt.ylabel('Predicted Energy Consumption')
    plt.tight_layout()
    plt.savefig("./static/energy_plot.png", dpi=300, bbox_inches='tight')

    # Linear+Ridge+Lasso regression for stability comparison
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1)
    }
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name} - Mean Squared Error: {mse}, R-squared: {r2}")
        
        # Visualization for Ridge and Lasso
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test.values, y_pred)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.title(f'Actual vs Predicted Energy Consumption ({name})')
        plt.xlabel('Actual Energy Consumption')
        plt.ylabel('Predicted Energy Consumption')
        plt.tight_layout()
        plt.savefig(f"./static/energy_plot_{name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')

    # remove world rows from country column for Pareto/Lorenz
    df7 = df4[df4['Country'] != 'World']
    cumulative_energy = df7.groupby('Country')['Energy_consumption'].sum().sort_values(ascending=False).cumsum().head(30)
    total_energy = cumulative_energy.iloc[-1]
    cumulative_percentage = cumulative_energy / total_energy * 100
    
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_percentage.index, cumulative_percentage.values, marker='o')
    plt.title('Cumulative Energy Consumption by Country')
    plt.xlabel('Country')
    plt.ylabel('Cumulative Percentage of Energy Consumption')
    plt.xticks(rotation=90)
    plt.grid()
    plt.tight_layout()
    plt.savefig("./static/Cumulative Energy Consumption by Country.png", dpi=300, bbox_inches='tight')

    energy_values = df7.groupby('Country')['Energy_consumption'].sum().values
    gini = gini_coefficient(energy_values)
    print(f"Gini Coefficient for Energy Consumption: {gini}")

    # Lorenz curve
    energy_sorted = np.sort(energy_values)
    cumulative_energy_plot = np.cumsum(energy_sorted)
    cumulative_energy_percentage = cumulative_energy_plot / cumulative_energy_plot[-1]
    population_percentage = np.arange(1, len(energy_sorted) + 1) / len(energy_sorted)
    
    plt.figure(figsize=(10, 6))
    plt.plot(population_percentage, cumulative_energy_percentage, label='Lorenz Curve', color='blue')
    plt.plot([0, 1], [0, 1], label='Line of Equality', color='red', linestyle='--')
    plt.fill_between(population_percentage, cumulative_energy_percentage, population_percentage, color='lightblue', alpha=0.5, label=f'Gini Coefficient = {gini:.2f}')
    plt.title('Lorenz Curve for Energy Consumption')
    plt.xlabel('Cumulative Share of Countries')
    plt.ylabel('Cumulative Share of Energy Consumption')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("./static/Lorenz Curve for Energy Consumption.png", dpi=300, bbox_inches='tight')

    # pareto cutoff analysis
    pareto_cutoff = cumulative_percentage[cumulative_percentage <= 80].index
    print("Countries contributing to 80% of energy consumption:\n", pareto_cutoff)

    # gini coefficient interpretation over time
    years = df7['Year'].unique()
    gini_over_time = {}
    for year in years:
        energy_values_year = df7[df7['Year'] == year].groupby('Country')['Energy_consumption'].sum().values
        gini_year = gini_coefficient(energy_values_year)
        gini_over_time[year] = gini_year
        
    plt.figure(figsize=(10, 6))
    plt.plot(list(gini_over_time.keys()), list(gini_over_time.values()), marker='o')
    plt.title('Gini Coefficient of Energy Consumption Over Time')
    plt.xlabel('Year')
    plt.ylabel('Gini Coefficient')
    plt.grid()
    plt.tight_layout()
    plt.savefig("./static/Gini Coefficient of Energy Consumption Over Time.png", dpi=300, bbox_inches='tight')

    # Create a summary dataframe for the web dashboard
    summary_df = df4.groupby(['Country', 'Year'])[['Energy_consumption', 'Population', 'GDP']].mean().reset_index()
    summary_df.to_csv("energy_data_web.csv", index=False)

    # Export Gini data
    gini_df = pd.DataFrame(list(gini_over_time.items()), columns=['Year', 'Gini_Coefficient'])
    gini_df.to_csv("gini_trends.csv", index=False)
    print("CSV files exported for Web Dashboard.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)