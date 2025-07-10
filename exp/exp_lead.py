
from exp.exp_main import Exp_Main
from models.LIFT import Model as LIFT
import os

import warnings
warnings.filterwarnings('ignore')


class Exp_Lead(Exp_Main):
    def __init__(self, args):
        super(Exp_Lead, self).__init__(args)

    def _build_model(self, model=None, framework_class=None):
        if framework_class is None:
            framework_class = LIFT
        model = super()._build_model(model, framework_class=framework_class)
        return model

    def _get_data(self, flag, **kwargs):
        # Add LIFT-specific data loading parameters
        if hasattr(self.args, 'lift') and self.args.lift:
            prefetch_path = os.path.join(self.args.prefetch_path, f'{flag}.npz')
            kwargs.update({
                'prefetch_path': prefetch_path,
                'batch_size': self.args.prefetch_batch_size if flag == 'train' else self.args.batch_size,
                'leader_num': self.args.leader_num,
                'state_num': self.args.state_num
            })
        return super()._get_data(flag, **kwargs)
