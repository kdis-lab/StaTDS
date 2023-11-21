import matplotlib.pyplot as plt
import numpy as np
import stats


def qq_plot(data: np.array):
    theoretical_quantiles = np.sort(np.random.normal(size=len(data)))
    data_quantiles = np.sort(data)

    plt.figure(figsize=(8, 8), facecolor='none')
    plt.plot(theoretical_quantiles, data_quantiles, 'o')
    plt.plot([min(theoretical_quantiles), max(theoretical_quantiles)],
             [min(theoretical_quantiles), max(theoretical_quantiles)], 'r--')
    plt.xlabel('Theorical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.title('QQ Plot')
    plt.grid(True)

    return plt.gcf()


def pp_plot(data: np.array):
    sorted_data = np.sort(data)
    empirical_cdf = np.arange(1, len(data) + 1) / len(data)
    theoretical_cdf = [stats.get_cdf_normal(value) for value in sorted_data]

    plt.figure(figsize=(8, 8), facecolor='none')
    plt.scatter(theoretical_cdf, empirical_cdf)
    plt.plot([0, 1], [0, 1], color='red')  # Línea de referencia
    plt.xlabel('Empirical CDF')
    plt.ylabel('Theoretical CDF (Normal)')
    plt.title('PP Plot')
    plt.grid(True)

    return plt.gcf()
