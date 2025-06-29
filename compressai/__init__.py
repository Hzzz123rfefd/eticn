from compressai.model import ETICN, ETICNQVRF, ETICNCQVR, STF, VIC, VAIC
from compressai.model_vbr import VIC_CQVR, VAIC_CQVR, STF_CQVR, STF_QVRF, VAIC_QVRF, VIC_QVRF
from compressai.dataset import DatasetForETICN, DatasetForImageCompression


datasets = {
    "eticn":DatasetForETICN,
    "compression": DatasetForImageCompression
}

models = {
    "eticn": ETICN,
    "stf": STF,
    "vic":VIC,
    "vaic": VAIC,
    
    "eticnqvrf": ETICNQVRF,
    "stfqvrf": STF_QVRF, 
    "vicqvrf": VIC_QVRF,
    "vaicqvrf": VAIC_QVRF,
    
    "viccqvr": VIC_CQVR,
    "vaiccqvr": VAIC_CQVR,
    "stfcqvr": STF_CQVR,
    "eticncqvr": ETICNCQVR
}


_entropy_coder = "ans"
_available_entropy_coders = [_entropy_coder]

try:
    import range_coder

    _available_entropy_coders.append("rangecoder")
except ImportError:
    pass


def set_entropy_coder(entropy_coder):
    """
    Specifies the default entropy coder used to encode the bit-streams.

    Use :mod:`available_entropy_coders` to list the possible values.

    Args:
        entropy_coder (string): Name of the entropy coder
    """
    global _entropy_coder
    if entropy_coder not in _available_entropy_coders:
        raise ValueError(
            f'Invalid entropy coder "{entropy_coder}", choose from'
            f'({", ".join(_available_entropy_coders)}).'
        )
    _entropy_coder = entropy_coder


def get_entropy_coder():
    """
    Return the name of the default entropy coder used to encode the bit-streams.
    """
    return _entropy_coder


def available_entropy_coders():
    """
    Return the list of available entropy coders.
    """
    return _available_entropy_coders
