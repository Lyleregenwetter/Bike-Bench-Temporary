from numpy import array
from torch.autograd import Variable
from torch.optim import SGD
from torch import Tensor
from ...util_global.constant import scalar_dict, solution_eps, get_hv_ref_dict
import torch
from tqdm import tqdm
from pymoo.indicators.hv import HV


class GradBaseSolver:
    def __init__(self, step_size, max_iter, tol):
        self.step_size = step_size
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, problem, x, prefs, args):
        '''
            :param problem:
            :param x:
            :param agg:
            :return:
                is a dict with keys: x, y
        '''

        # The abstract class cannot be implemented directly.
        raise NotImplementedError



class GradAggSolver(GradBaseSolver):
    def __init__(self, step_size, max_iter, tol, device='cpu'):
        self.device = device
        super().__init__(step_size, max_iter, tol)

    def solve(self, problem, x, prefs, args, ref_point):
        x = torch.tensor(x, dtype=torch.float32, requires_grad=True, device=self.device)

        # ref_point = array([2.0, 2.0])
        # ind = HV(ref_point = get_hv_ref_dict(args.problem_name))
        # ind = HV(ref_point = array([1.0, 1.0]))
        
        # ind = HV(ref_point = ref_point)
    



        # hv_arr = []
        y_arr = []

        if not isinstance(prefs, torch.Tensor):
            prefs = torch.tensor(prefs, dtype=torch.float32, device=self.device)
        else:
            prefs = prefs.to(dtype=torch.float32, device=self.device)

        # prefs = Tensor(prefs)
        optimizer = SGD([x], lr=self.step_size)
        agg_func = scalar_dict[args.agg]
        res = {}
        for i in tqdm(range(self.max_iter)):
            y = problem.evaluate(x)
            
            # hv_arr.append(ind.do(y.detach().cpu().numpy()))

            agg_val = agg_func(y, prefs)
            optimizer.zero_grad()
            torch.sum(agg_val).backward()
            optimizer.step()

            y_arr.append(y.detach().cpu().numpy())

            if 'lbound' in dir(problem):
                x.data = torch.clamp(x.data, 
                     torch.tensor(problem.lbound, device=x.device, dtype=torch.float32) + solution_eps, 
                     torch.tensor(problem.ubound, device=x.device, dtype=torch.float32) - solution_eps)


        res['x'] = x.detach().cpu().numpy()
        res['y'] = y.detach().cpu().numpy()
        # res['hv_arr'] = hv_arr
        res['y_arr'] = y_arr
        return res