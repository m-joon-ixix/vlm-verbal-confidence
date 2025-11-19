import os

DEFAULT_SEED = 42


def get_seed() -> int:
    seed_env = os.getenv("SEED")
    if seed_env:
        try:
            return int(seed_env)
        except ValueError:
            print(f"Invalid SEED env: {seed_env} => Using default seed: {DEFAULT_SEED}")
            return DEFAULT_SEED
    else:
        return DEFAULT_SEED
