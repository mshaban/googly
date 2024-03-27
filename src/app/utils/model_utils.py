from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np


def pad_vector(input_vector: np.ndarray, missing_samples: int) -> np.ndarray:
    """
    Pads the input vector with zeros to add missing samples.

    Parameters:
    input_vector (np.ndarray): The input vector to be padded.
    missing_samples (int): The number of missing samples to add to the input vector.

    Returns:
    np.ndarray: The padded input vector with missing samples added.

    Raises:
    ValueError: If input_vector is not a numpy array.
    ValueError: If missing_samples is not a positive integer.
    """

    if not isinstance(input_vector, np.ndarray):
        raise ValueError("input_vector must be a numpy array")
    if not isinstance(missing_samples, int) or missing_samples <= 0:
        raise ValueError("missing_samples must be a positive integer")

    pad_width = ((missing_samples, 0), (0, 0), (0, 0), (0, 0))
    return np.pad(input_vector, pad_width, mode="constant", constant_values=0)


def batch_vector_generator(input_vector: np.ndarray, batch_size: int):

    n_samples = input_vector.shape[0]

    if n_samples == batch_size:
        yield input_vector
    elif n_samples < batch_size:
        yield pad_vector(input_vector, batch_size - n_samples)
    else:
        for i in range(0, n_samples, batch_size):
            end_index = i + batch_size
            if end_index > n_samples:
                yield pad_vector(input_vector[i:], batch_size - (end_index - n_samples))
            else:
                yield input_vector[i:end_index]


def batch_process(
    items: list[Any], process_func: Callable, max_workers: int = 4
) -> list[Any]:
    """
    Processes a list of items in parallel using the given function.

    Args:
        items (List[Any]): The list of input items to be processed. Each item
        could be of any type.
        process_func (Callable): The function to apply to each item. This
        function should be capable of handling the types of items in the list.
        max_workers (int): The maximum number of workers to use for parallel
        processing. Default is 4.

    Returns:
        List[Any]: The list of processed items.

    Raises:
        ValueError: If the items parameter is not a list.
    """

    if not isinstance(items, list):
        raise ValueError("The items parameter must be a list.")

    def process_item(item):
        if isinstance(item, tuple):
            return process_func(*item)  # Unpack tuple and pass as multiple arguments
        else:
            return process_func(item)  # Pass item directly

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_item, item) for item in items]
        processed_items = [future.result() for future in futures]

    return processed_items
