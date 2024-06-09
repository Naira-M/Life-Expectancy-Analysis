import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.stats.anova import anova_lm


def perform_regression(df):
    # Define the independent variables (X) and the dependent variable (y)
    X = df.drop('Life expectancy at birth', axis=1)  # Independent variables
    y = df['Life expectancy at birth']  # Dependent variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Add a constant to the independent variables (for intercept term)
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    model = sm.OLS(y_train, X_train).fit()
    print(model.summary())
    
    y_pred = model.predict(X_test)

    # Plot the actual vs. predicted values
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Plot the regression line
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title('Actual vs. Predicted values')
    plt.show()

    return model

def perform_anova_test(df):
    # Prepare the data
    X_all = df.drop('Life expectancy at birth', axis=1)
    X_without_unemployment = df.drop(['Life expectancy at birth', 'Unemployment'], axis=1)
    y = df['Life expectancy at birth']
    # Add constant for intercept
    X_all = sm.add_constant(X_all)
    X_without_unemployment = sm.add_constant(X_without_unemployment)
    # Fit the models
    model_all = sm.OLS(y, X_all).fit()
    model_without_unemployment = sm.OLS(y, X_without_unemployment).fit()
    # Perform ANOVA
    
    anova_results = anova_lm(model_without_unemployment, model_all)
    return anova_results
    