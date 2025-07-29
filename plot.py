import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, expon, gamma, beta, t, chi2

# Disable the warning
st.set_option('deprecation.showPyplotGlobalUse', False)


def plot_distribution(distribution, params):
    """Plots the specified probability distribution."""
    x = np.linspace(params['min'], params['max'], 1000)  # x-axis of the plot - Creates an array of 1000 evenly spaced points between the minimum and maximum values.

    # Checking the distribution and calculating y-values
    # Calculates the probability density function (PDF) for the selected distribution
    if distribution == 'Normal':
        y = norm.pdf(x, params['mean'], params['std'])
        label = f"Normal Distribution (μ={params['mean']}, σ={params['std']})"
    elif distribution == 'Uniform':
        y = uniform.pdf(x, loc=params['a'], scale=params['b'])
        label = f"Uniform Distribution (a={params['a']}, b={params['b']})"
    elif distribution == 'Exponential':
        y = expon.pdf(x, scale=params['lambda'])
        label = f"Exponential Distribution (λ={params['lambda']})"
    elif distribution == 'Gamma':
        y = gamma.pdf(x, a=params['alpha'], scale=params['betaa'])
        label = f"Gamma Distribution (α={params['alpha']}, β={params['betaa']})"
    elif distribution == 'Beta':
        y = beta.pdf(x, a=params['alpha'], b=params['betaa'])
        label = f"Beta Distribution (α={params['alpha']}, β={params['betaa']})"
    elif distribution == "Student's t":
        y = t.pdf(x, df=params['df'])
        label = f"Student's t Distribution (df={params['df']})"
    elif distribution == 'Chi-Squared':
        y = chi2.pdf(x, df=params['df'])
        label = f"Chi-Squared Distribution (df={params['df']})"
    else:
        raise ValueError(f"Invalid distribution: {distribution}")

    # Creates a plot of the PDF, labels the axes, and adds a title.
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=label)
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title(f"{distribution} Distribution")

    # Shade the specified region
    if distribution == 'Normal':
        plt.fill_between(x, y, where=(x >= params['lower_bound']) & (x <= params['upper_bound']), alpha=0.2)
    elif distribution == 'Uniform':
        plt.fill_between(x, y, where=(x >= params['lower_bound']) & (x <= params['upper_bound']), alpha=0.2)
    elif distribution == 'Exponential':
        plt.fill_between(x, y, where=(x >= params['lower_bound']) & (x <= params['upper_bound']), alpha=0.2)
    elif distribution == 'Gamma':
        plt.fill_between(x, y, where=(x >= params['lower_bound']) & (x <= params['upper_bound']), alpha=0.2)
    elif distribution == 'Beta':
        plt.fill_between(x, y, where=(x >= params['lower_bound']) & (x <= params['upper_bound']), alpha=0.2)
    elif distribution == "Student's t":
        plt.fill_between(x, y, where=(x >= params['lower_bound']) & (x <= params['upper_bound']), alpha=0.2)
    elif distribution == 'Chi-Squared':
        plt.fill_between(x, y, where=(x >= params['lower_bound']) & (x <= params['upper_bound']), alpha=0.2)

    plt.legend()
    st.pyplot()  # to display the plot in the streamlit app

# Streamlit app:
def main():
    st.title("Probability Distributions: Plot & Calculate")
    
    # Dropdown menu for selecting the distribution
    distribution = st.selectbox("Select a distribution:", ["Normal", "Uniform", "Exponential", "Gamma", "Beta","Student's t","Chi-Squared"])
    
    # Input parameters
    if distribution == 'Normal':
        mean = st.number_input("Mean (μ):", value=0.0)
        std = st.number_input("Standard Deviation (σ):", value=1.0)
        min_val = st.number_input("Minimum x-value:", value=-3.0)
        max_val = st.number_input("Maximum x-value:", value=3.0)
        lower_bound = st.number_input("Lower Bound:", value=-2.0)
        upper_bound = st.number_input("Upper Bound:", value=2.0)
        params = {'distribution': distribution, 'mean': mean, 'std': std, 'min': min_val, 'max': max_val, 'lower_bound': lower_bound, 'upper_bound': upper_bound}
    elif distribution == 'Uniform':
        a = st.number_input("Minimum value (a):", value=0.0)
        b = st.number_input("Maximum value (b):", value=1.0)
        min_val = st.number_input("Minimum x-value:", value=a)
        max_val = st.number_input("Maximum x-value:", value=b)
        lower_bound = st.number_input("Lower Bound:", value=a)
        upper_bound = st.number_input("Upper Bound:", value=b)
        params = {'distribution': distribution, 'a': a, 'b': b, 'min': min_val, 'max': max_val, 'lower_bound': lower_bound, 'upper_bound': upper_bound}
    elif distribution == 'Exponential':
        lambda_val = st.number_input("Rate parameter (λ):", value=1.0)
        min_val = st.number_input("Minimum x-value:", value=0)
        max_val = st.number_input("Maximum x-value:", value=10)
        lower_bound = st.number_input("Lower Bound:", value=0.0)
        upper_bound = st.number_input("Upper Bound:", value=10.0)
        params = {'distribution': distribution, 'lambda': lambda_val, 'min': min_val, 'max': max_val, 'lower_bound': lower_bound, 'upper_bound': upper_bound}
    elif distribution == 'Gamma':
        alpha = st.number_input("Shape parameter (α):", value=2.0)
        betaa = st.number_input("Scale parameter (β):", value=1.0)
        min_val = st.number_input("Minimum x-value:", value=0)
        max_val = st.number_input("Maximum x-value:", value=10)
        lower_bound = st.number_input("Lower Bound:", value=0.0)
        upper_bound = st.number_input("Upper Bound:", value=10.0)
        params = {'distribution': distribution, 'alpha': alpha, 'betaa': betaa, 'min': min_val, 'max': max_val, 'lower_bound': lower_bound, 'upper_bound': upper_bound}
    elif distribution == 'Beta':
        alpha = st.number_input("Alpha parameter (α):", value=2.0)
        betaa = st.number_input("Beta parameter (β):", value=2.0)
        min_val = st.number_input("Minimum x-value:", value=0.0)
        max_val = st.number_input("Maximum x-value:", value=1.0)
        lower_bound = st.number_input("Lower Bound:", value=0.0)
        upper_bound = st.number_input("Upper Bound:", value=1.0)
        params = {'distribution': distribution, 'alpha': alpha, 'betaa': betaa, 'min': min_val, 'max': max_val, 'lower_bound': lower_bound, 'upper_bound': upper_bound}
    elif distribution == "Student's t":
        df = st.number_input("Degrees of Freedom (df):", value=10)
        min_val = st.number_input("Minimum x-value:", value=-3.0)
        max_val = st.number_input("Maximum x-value:", value=3.0)
        lower_bound = st.number_input("Lower Bound:", value=-2.0)
        upper_bound = st.number_input("Upper Bound:", value=2.0)
        params = {'distribution': distribution, 'df': df, 'min': min_val, 'max': max_val, 'lower_bound': lower_bound, 'upper_bound': upper_bound}
    elif distribution == 'Chi-Squared':
        df = st.number_input("Degrees of Freedom (df):", value=10)
        min_val = st.number_input("Minimum x-value:", value=0.0)
        max_val = st.number_input("Maximum x-value:", value=10.0)
        lower_bound = st.number_input("Lower Bound:", value=0.0)
        upper_bound = st.number_input("Upper Bound:", value=10.0)
        params = {'distribution': distribution, 'df': df, 'min': min_val, 'max': max_val, 'lower_bound': lower_bound, 'upper_bound': upper_bound}

    if st.button("Plot Distribution"):
        plot_distribution(params['distribution'], params)
        
        # Calculates the probability of the specified interval using the cumulative distribution function (CDF) of the selected distribution.
        # Calculate and display the probability
        if distribution == 'Normal':
            probability = norm.cdf(upper_bound, params['mean'], params['std']) - norm.cdf(lower_bound, params['mean'], params['std'])
        elif distribution == 'Uniform':
            probability = uniform.cdf(upper_bound, loc=params['a'], scale=params['b']) - uniform.cdf(lower_bound, loc=params['a'], scale=params['b'])
        elif distribution == 'Exponential':
            probability = expon.cdf(upper_bound, scale=params['lambda']) - expon.cdf(lower_bound, scale=params['lambda'])
        elif distribution == 'Gamma':
            probability = gamma.cdf(upper_bound, a=params['alpha'], scale=params['betaa']) - gamma.cdf(lower_bound, a=params['alpha'], scale=params['betaa'])
        elif distribution == 'Beta':
            probability = beta.cdf(upper_bound, a=params['alpha'], b=params['betaa']) - beta.cdf(lower_bound, a=params['alpha'], b=params['betaa'])
        elif distribution == "Student's t":
            probability = t.cdf(upper_bound, df=params['df']) - t.cdf(lower_bound, df=params['df'])
        elif distribution == 'Chi-Squared':
            probability = chi2.cdf(upper_bound, df=params['df']) - chi2.cdf(lower_bound, df=params['df'])

        st.write(f"Probability: {probability:.4f}")

if __name__ == '__main__':
    main()