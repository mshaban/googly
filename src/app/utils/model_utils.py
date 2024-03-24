from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import numpy as np


def pad_object_list(object_list: list, target_length: int) -> list:
    """
    Pads a given list of objects with additional items to reach a target length.

    Args:
    object_list (list): The list of objects to be padded.
    target_length (int): The desired length of the padded list.

    Returns:
    list: A new list containing the original objects with additional items added to reach the target length.

    Raises:
    ValueError: If the target length is less than the current length of the object list.

    Example:
    >>> pad_object_list([1, 2, 3], 5)
    [1, 2, 3, None, None]
    """

    current_length = len(object_list)
    if current_length >= target_length:
        return object_list
    additional_items_needed = target_length - current_length
    additional_items = [None for i in range(additional_items_needed)]
    padded_list = object_list + additional_items
    return padded_list


def pad_vector(input_vector: np.ndarray, missing_samples: int) -> np.ndarray:
    """
    Pad a given input vector with zeros to add missing samples.

    Parameters:
    input_vector (np.ndarray): The input vector to be padded with zeros.
    missing_samples (int): The number of missing samples to add to the input vector.

    Returns:
    np.ndarray: The padded input vector with missing samples added.

    Raises:
    ValueError: If input_vector is not a numpy array or if missing_samples is not a positive integer.
    """

    if not isinstance(input_vector, np.ndarray):
        raise ValueError("input_vector must be a numpy array")
    if not isinstance(missing_samples, int) or missing_samples < 0:
        raise ValueError("missing_samples must be a positive integer")

    if missing_samples == 0:
        return input_vector

    pad_width = ((missing_samples, 0), (0, 0), (0, 0), (0, 0))
    return np.pad(input_vector, pad_width, mode="constant", constant_values=0)


def duplicate_samples(input_vector: np.ndarray, total_samples: int) -> np.ndarray:
    """
    Duplicate the input vector to generate a specified number of total samples.

    Parameters:
    input_vector (np.ndarray): The input vector to be duplicated.
    total_samples (int): The total number of samples to generate.

    Returns:
    np.ndarray: The duplicated vector with the specified number of total samples.

    Raises:
    ValueError: If the input vector is empty.
    """

    n_samples = input_vector.shape[0]

    if n_samples == 0:
        raise ValueError("Input vector cannot be empty.")

    if n_samples >= total_samples:
        return input_vector
    else:
        repeats = (total_samples + n_samples - 1) // n_samples
        duplicated = np.tile(input_vector, (repeats, 1, 1, 1))
        return duplicated[:total_samples]


def batch_vector_generator(
    input_vector: np.ndarray, batch_size: int, duplicate_pad: bool = True
):
    """
    Generate batches of vectors from an input vector.

    Parameters:
    input_vector (np.ndarray): The input vector from which batches will be generated.
    batch_size (int): The size of each batch.
    duplicate_pad (bool, optional): Flag to indicate whether to duplicate samples to fill the last batch. Default is True.

    Yields:
    np.ndarray: A batch of vectors of size batch_size.

    Raises:
    ValueError: If batch_size is not a positive integer.
    """

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    n_samples = input_vector.shape[0]

    if n_samples <= batch_size:
        # Use the duplicate_samples function to fill the batch
        yield duplicate_samples(input_vector, batch_size)
    else:
        for i in range(0, n_samples, batch_size):
            end_index = i + batch_size
            if end_index > n_samples and duplicate_pad:
                # For the last batch, if it's smaller than the batch_size, duplicate samples
                yield duplicate_samples(input_vector[i:], batch_size)
            else:
                yield input_vector[i:end_index]


def batch_process(
    items: list[Any], process_func: Callable, max_workers: int = 4
) -> list[Any]:
    """
    Process a list of items using a given function in parallel using ThreadPoolExecutor.

    Args:
        items (list): A list of items to be processed.
        process_func (Callable): A function that will be applied to each item in the list.
        max_workers (int, optional): The maximum number of worker threads to use for processing. Defaults to 4.

    Returns:
        list: A list of processed items.

    Raises:
        ValueError: If the items parameter is not a list.

    Example:
        >>> def square(x):
        ...     return x ** 2
        >>> batch_process([1, 2, 3, 4], square)
        [1, 4, 9, 16]
    """
    if not isinstance(items, list):
        raise ValueError("The items parameter must be a list.")

    def process_item(item):
        if isinstance(item, tuple):
            return process_func(*item)
        else:
            return process_func(item)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_item, item) for item in items]
        processed_items = [future.result() for future in futures]

    return processed_items
