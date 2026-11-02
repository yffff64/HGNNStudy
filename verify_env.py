import warnings; warnings.filterwarnings('ignore')
import torch, numpy, scipy, yaml, dask, ipdb, torch_geometric
import torch_scatter, torch_sparse, torch_cluster, torch_spline_conv, deeprobust
lines = []
lines.append('torch %s cuda=%s %s' % (torch.__version__, torch.cuda.is_available(),
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''))
lines.append('numpy %s' % numpy.__version__)
lines.append('scipy %s' % scipy.__version__)
lines.append('pyg %s' % torch_geometric.__version__)
lines.append('scatter %s sparse %s' % (getattr(torch_scatter,'__version__','ok'), getattr(torch_sparse,'__version__','ok')))
lines.append('cluster %s spline %s' % (getattr(torch_cluster,'__version__','ok'), getattr(torch_spline_conv,'__version__','ok')))
lines.append('deeprobust %s dask %s yaml %s' % (getattr(deeprobust,'__version__','ok'), dask.__version__, yaml.__version__))
lines.append('ALL IMPORTS OK')
open(r'C:\Users\y\'f\Desktop\DHG-Bench-main\DHG-Bench-main\verify_out.txt','w',encoding='utf-8').write('\n'.join(lines))
