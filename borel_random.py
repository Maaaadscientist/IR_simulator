import numpy as np
import scipy

def borel_pmf(k, lambda_):
    # Use logarithms for numerical stability
    log_pmf = (k - 1) * np.log(lambda_ * k) - k * lambda_ - scipy.special.gammaln(k + 1)
    return np.exp(log_pmf)
#
#def borel_pmf(k, lambda_):
#    return (lambda_ * k)**(k - 1) * np.exp(-k * lambda_) / math.factorial(k)


def generate_random_borel(lambda_, max_k=32):
    # Generate PMF values
    pmf_values = [borel_pmf(k+1, lambda_) for k in range(max_k)]

    # Normalize the PMF
    total = sum(pmf_values)
    normalized_pmf = [value / total for value in pmf_values]

    # Random sampling based on the PMF
    return np.random.choice(range(max_k), p=normalized_pmf)

