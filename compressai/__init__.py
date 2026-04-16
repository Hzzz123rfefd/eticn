from compressai.model import ETICN, STF, JAHP, MLIC, ELIC, NetA, NetB, NetC
from compressai.model_vbr import (
    ETICN_QEEVRF, ETICN_QVRF, ETICN_AGVAE, ETICN_STVRF, ETICN_MSD,
    STF_QEEVRF, STF_QVRF, STF_AGVAE, STF_STVRF, STF_MSD,
    JAHP_QEEVRF, JAHP_QVRF, JAHP_AGVAE,JAHP_STVRF, JAHP_MSD,
)
from compressai.dataset import DatasetForETICN, DatasetForImageCompression


datasets = {
    "eticn":DatasetForETICN,
    "compression": DatasetForImageCompression
}

models = {
    "eticn": ETICN,
    "mlic": MLIC,
    "stf": STF,
    "jahp": JAHP,
    "elic": ELIC,
    "neta": NetA,
    "netb": NetB,
    "netc": NetC,
    
    "jahpmsd": JAHP_MSD,
    "stfmsd": STF_MSD,
    "eticnmsd": ETICN_MSD,
    
    "eticnqvrf": ETICN_QVRF,
    "stfqvrf": STF_QVRF, 
    "jahpqvrf": JAHP_QVRF,
    
    "jahpqeevrf": JAHP_QEEVRF,
    "stfqeevrf": STF_QEEVRF,
    "eticnqeevrf": ETICN_QEEVRF,
    
    "eticnagvae": ETICN_AGVAE,
    "stfagvae": STF_AGVAE,
    "jahpagvae": JAHP_AGVAE,
    
    "eticnstvrf": ETICN_STVRF, 
    "stfstvrf": STF_STVRF,
    "jahpstvrf": JAHP_STVRF
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
