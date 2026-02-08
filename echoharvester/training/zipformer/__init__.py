"""Vendored Zipformer2 encoder from k2-fsa/icefall (Apache 2.0).

Source: https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/zipformer
Files: scaling.py, subsampling.py, zipformer.py

Modifications:
- Import paths changed from relative to echoharvester.training.zipformer.*
- EncoderInterface removed (nn.Module direct inheritance)
- icefall.utils.torch_autocast replaced with local shim
- k2 dependency made optional (falls back to pure-torch SwooshL/R)
"""

from echoharvester.training.zipformer.subsampling import Conv2dSubsampling
from echoharvester.training.zipformer.zipformer import Zipformer2

__all__ = ["Zipformer2", "Conv2dSubsampling"]
