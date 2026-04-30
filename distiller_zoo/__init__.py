from .KD import DistillKL
from .FitNet import HintLoss
from .AT import Attention
from .SP import Similarity
from .VID import VIDLoss
from .SemCKD import SemCKDLoss
from .AFD import AFD
from .AFD_improved import AFD as AFD_improved

__all__ = ['DistillKL', 'HintLoss', 'Attention', 'Similarity', 'VIDLoss', 'SemCKDLoss', 'AFD', 'AFD_improved']
