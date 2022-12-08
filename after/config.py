from dataclasses import dataclass

@dataclass
class Files:
    file_name: str

@dataclass(frozen=True)
class Paths:
    data_dir: str

@dataclass
class Runparams:
    test_size: float
    random_state: int

@dataclass
class Knnparams:
    n_init: int
    n_clusters: int
    max_iter: int
    tol: float
    algorithm: str

@dataclass
class MainConfig:
    files: Files
    paths: Paths
    runparams: Runparams
    knnparams: Knnparams
