from typing import Optional

import numpy as np


def np_random(seed: Optional[int] = None) -> np.random.Generator:
    """Generates a random number generator from the seed and returns the Generator and
    seed.

    Full credit to OpenAI implementation at
    https://github.com/openai/gym/blob/master/gym/utils/seeding.py.


    Parameters
    ----------
    seed : int, optional
        The seed used to create the generator.

    Returns
    -------
    Generator
        The generator initialized with the given seed.

    Raises
    ------
    ValueError
        Seed must be a non-negative integer or omitted.
    """
    if seed is not None and not (isinstance(seed, int) and seed >= 0):
        raise ValueError(f"Seed must be a non-negative integer or omitted, not {seed}.")
    seed_seq = np.random.SeedSequence(seed)
    return np.random.Generator(np.random.PCG64(seed_seq))
