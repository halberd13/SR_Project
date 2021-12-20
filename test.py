import os
import torch
from torch import nn
from torch.autograd import Function

BASICSR_JIT = os.getenv('BASICSR_EXT')
# if BASICSR_JIT == 'True':
    
#     from torch.utils.cpp_extension import load
#     module_path = os.path.dirname(__file__)
#     fused_act_ext = load(
#         'fused',
#         sources=[
#             os.path.join(module_path, 'src', 'fused_bias_act.cpp'),
#             os.path.join(module_path, 'src', 'fused_bias_act_kernel.cu'),
#         ],
#     )
# else:
try:
    from . import fused_act_ext
    from torch.utils.cpp_extension import load
    module_path = os.path.dirname(__file__)
    print(__file__)
    fused_act_ext = load(
        'fused',
        sources=[
            os.path.join(module_path, 'src', 'fused_bias_act.cpp'),
            os.path.join(module_path, 'src', 'fused_bias_act_kernel.cu'),
        ],
    )
    print(module_path)
except ImportError:
    pass
    # avoid annoying print output
    # print(f'Cannot import deform_conv_ext. Error: {error}. You may need to: \n '
    #       '1. compile with BASICSR_EXT=True. or\n '
    #       '2. set BASICSR_JIT=True during running')