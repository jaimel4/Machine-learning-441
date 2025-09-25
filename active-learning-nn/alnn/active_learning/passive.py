from typing import Sequence, Set

def query_passive(unlabeled: Set[int], k: int) -> Sequence[int]:
    """Random sampling from the unlabeled pool."""
    return list(sorted(list(unlabeled)))[:k]
