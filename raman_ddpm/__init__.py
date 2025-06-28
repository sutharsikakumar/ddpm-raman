"""
raman_ddpm package initializer
──────────────────────────────
Re-exports the public classes / helpers so user code can simply do:

    from raman_ddpm import RamanDataset, get_loaders, RamanCNN, CNNTrainer, …

Any legacy reference to `discover_labels` has been removed because
the new balanced loader embeds class discovery internally.
"""


from .dataset import RamanDataset, get_loaders


from .ddpm_model   import Diffusion1D
from .ddpm_trainer import DDPMTrainer
from .sampler      import sample_spectra


from .cnn_model    import RamanCNN
from .cnn_trainer  import CNNTrainer

__all__ = [
    "RamanDataset", "get_loaders",
    "Diffusion1D", "DDPMTrainer", "sample_spectra",
    "RamanCNN",    "CNNTrainer",
]
