from collections.abc import Iterable, Callable
from scipy.stats import Normal
import numpy as np

class MCSampler:
    def __init__(self, seed: int = None):
        """
        Initializes a Monte Carlo (MC) sample generator for uniform distributions in [0, 1] with an optional random seed.
        If no seed is provided, the generator will be initialized with a random seed.

        Args:
            seed (int, optional): The seed for the random number generator. 
                If None, the generator will be initialized with a random seed. 
                Defaults to None.
        """
        self.rng = np.random.default_rng(seed)

    def __call__(self, nsamples: int, nfactors: int) -> np.ndarray:
        """
        Generates a sample of size (nsamples, nfactors) from a uniform distribution in [0, 1].
        The samples are generated using the random number generator initialized with the provided seed.

        Args:
            nsamples (int): The number of samples to generate.
            nfactors (int): The number of factors (features) for each sample.

        Returns:
            np.ndarray: A 2D array of shape (nsamples, nfactors) containing the generated samples.
        """
        return self.rng.random(size=(nsamples, nfactors))
    
class LHSampler:
    def __init__(self, seed: int = None):
        """
        Initializes a Latin Hypercube (LH) sample generator for uniform distributions in [0, 1] with an optional random seed.
        If no seed is provided, the generator will be initialized with a random seed.

        Args:
            seed (int, optional): The seed for the random number generator.
                If None, the generator will be initialized with a random seed.
                Defaults to None.
        """
        self.rng = np.random.default_rng(seed)

    def __call__(self, nsamples: int, nfactors: int) -> np.ndarray:
        """
        Generates a sample of size (nsamples, nfactors) from a uniform distribution in [0, 1] using Latin Hypercube Sampling.
        The samples are generated using the random number generator initialized with the provided seed.

        Args:
            nsamples (int): The number of samples to generate.
            nfactors (int): The number of factors (features) for each sample.

        Returns:
            np.ndarray: A 2D array of shape (nsamples, nfactors) containing the generated samples.
        """
        interval_bounds = np.linspace(0,1,nsamples+1)
        intervals = self.rng.permuted(np.array(nfactors*np.arange(nsamples).tolist()).reshape((nsamples,nfactors),order="F"), axis=0)
        return self.rng.uniform(interval_bounds[intervals],interval_bounds[intervals+1])
    
class SampleGenerator:
    def __init__(self, uniformSampler: Callable):
        """
        Args:
            uniformSampler (Callable): Callable that provides the __call__-method with the arguments nsamples and nfeatures returning a np.ndarray with the shape (nsamples, nfeatures)
        """
        self.sampler = uniformSampler

    def __call__(self, nsamples: int, mean: float | Iterable = 0, deviation: float | Iterable = 1, distribution: str | Iterable = "uniform", nfeatures: int = 1) -> np.ndarray:
        """
        Generate samples from a uniform or normal distribution with given mean and (standard) deviation.

        If mean, deviation or distribution are iterables, the number of features is inferred from their length.
        If none of them are iterables, nfeatures must be provided for more than one feature.

        Args:
            nsamples (int): The number of samples to generate.
            mean (float | Iterable, optional): The mean of the resulting distribution. Defaults to 0.
            deviation (float | Iterable, optional): The standard deviation of the resulting normal distribution or the maximal deviation of the resulting uniform distribution from its mean. Defaults to 1.
            distribution (str | Iterable, optional): The type of distribution to use for the transformation. Can be "uniform" or "normal". Defaults to "uniform".
            nfeatures (int | None, optional): The number of features to generate. Defaults to None.

        Returns:
            np.ndarray: The generated samples.
        """
        assert isinstance(nsamples, int), "nsamples must be an integer"

        mean_iterable = hasattr(mean, '__iter__')
        deviation_iterable = hasattr(deviation, '__iter__')
        distribution_iterable = hasattr(distribution, '__iter__') and not isinstance(distribution, str)

        if mean_iterable or deviation_iterable or distribution_iterable:
            lengths = []
            if mean_iterable:
                lengths.append(len(mean))
            if deviation_iterable:
                lengths.append(len(deviation))
            if distribution_iterable:
                lengths.append(len(distribution))

            assert len(set(lengths)) <= 1, "mean, deviation and distribution must have the same length or be scalars"
            nfeatures = lengths[0]

        assert isinstance(nfeatures, int), "nfeatures must be an integer"

        uniform_samples = self.sampler(nsamples, nfeatures)
        samples = np.zeros_like(uniform_samples)
        for i in range(nfeatures):
            current_mean = mean[i] if mean_iterable else mean
            current_deviation = deviation[i] if deviation_iterable else deviation
            current_distribution = distribution[i] if distribution_iterable else distribution
            samples[:, i] = self.transform_from_uniform(uniform_samples[:, i], current_mean, current_deviation, current_distribution)
        return samples
    
    def transform_from_uniform(self, x: float | np.ndarray, mean: float = 0, deviation: float = 1, distribution: str = "uniform") -> float | np.ndarray:
        """
        Transform a value or array of values from a uniform distribution in [0, 1] to a uniform or normal distribution with given mean and (standard) deviation.
        The resulting uniform distribution is in [mean - deviation, mean + deviation].

        Args:
            x (float | np.ndarray): The value(s) to transform.
            mean (float): The mean of the resulting distribution.
            deviation (float): The standard deviation of the resulting normal distribution or the maximal deviation of the resulting uniform distribution from its mean.
            distribution (str): The type of distribution to use for the transformation. Can be "uniform" or "normal".

        Returns:
            float | np.ndarray: The transformed value(s). If the input is a float, the output will be a float. If the input is an array, the output will be an array.
        """
        if distribution == "uniform":
            return mean + deviation * (x - 0.5) * 2
        elif distribution == "normal":
            return mean + deviation * Normal().icdf(x)
        raise ValueError(f"Invalid distribution type {distribution}. Use 'uniform' or 'normal'.")
    
    def transform_to_uniform(self, x: float | np.ndarray, mean: float = 0, deviation: float = 1, distribution: str = "uniform") -> float | np.ndarray:
        """
        Transform a value or array of values from a uniform or normal distribution with given mean and (standard) deviation to a uniform distribution in [0, 1].
        The prior uniform distribution is in [mean - deviation, mean + deviation].

        Args:
            x (float | np.ndarray): The value(s) to transform.
            mean (float): The mean of the prior distribution.
            deviation (float): The standard deviation of the prior normal distribution or the maximal deviation of the prior uniform distribution from its mean.
            distribution (str): The type of prior distribution to use for the transformation. Can be "uniform" or "normal".

        Returns:
            float | np.ndarray: The transformed value(s). If the input is a float, the output will be a float. If the input is an array, the output will be an array.
        """
        if distribution == "uniform":
            return 0.5 + (x - mean) / (2 * deviation)
        elif distribution == "normal":
            return Normal().cdf((x - mean) / deviation)
        raise ValueError(f"Invalid distribution type {distribution}. Use 'uniform' or 'normal'.")
    
    def transform(self, x: float | np.ndarray, mean_old: float = 0, deviation_old: float = 1, distribution_old: str = "uniform", mean_new: float = 0, deviation_new: float = 1, distribution_new: str = "uniform") -> float | np.ndarray:
        """
        Transform a value or array of values from one distribution to another.

        Args:
            x (float | np.ndarray): The value(s) to transform.
            mean_old (float): The mean of the prior distribution.
            deviation_old (float): The standard deviation of the prior normal distribution or the maximal deviation of the prior uniform distribution from its mean.
            distribution_old (str): The type of prior distribution to use for the transformation. Can be "uniform" or "normal".
            mean_new (float): The mean of the resulting distribution.
            deviation_new (float): The standard deviation of the resulting normal distribution or the maximal deviation of the resulting uniform distribution from its mean.
            distribution_new (str): The type of resulting distribution to use for the transformation. Can be "uniform" or "normal".

        Returns:
            float | np.ndarray: The transformed value(s). If the input is a float, the output will be a float. If the input is an array, the output will be an array.
        """
        return self.transform_from_uniform(self.transform_to_uniform(x, mean_old, deviation_old, distribution_old), mean_new, deviation_new, distribution_new)