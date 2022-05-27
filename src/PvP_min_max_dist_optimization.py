import numpy as np
from PvP_min_max_symm_objective import objective_function, prepare_matrix
N = 12
M = N - 1
dtype = np.float64
cnt = (M ** 2 - 1) // 4 - M // 2
bound = (0, 1)
bound_init = (0, 0.5)

strategy_center_flat = np.ones(cnt, dtype=dtype) * 0.5
strategy_border_flat = np.ones(cnt, dtype=dtype) * 0.5

def objective_function_except(*arg, **argw):
    try:
        fun = objective_function(*arg, **argw)
        if np.isnan(fun):
            print()
            print('BUG NAN!!!')
            print(arg, argw) 
            print(arg[0].dtype, arg[1].dtype, arg[0].shape, arg[1].shape)
            print(fun)
            print()
        return fun
    except Exception as e:
        print('BUG2!!!')
        print(e)
        print(arg, argw)
        return np.nan


from scipy.optimize import minimize
from tqdm import tqdm
from functools import partial
import time

progress_info = {
    'f_prev': None,
    'f': None,
    'mn': None,
    'mx': None,
    'x': None,
    'it': 0,
    't_start': time.time(),
    't_last': time.time(),
    't_delta': 0,
    't_ema': 0,
}

from multiprocessing import Process, Event, Lock
#from threading import Timer
import sys
lock_progress_info = Lock()

class Timer(Process):
    def __init__(self, interval, function, args=[], kwargs={}):
        super(Timer, self).__init__()
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.finished = Event()

    def cancel(self):
        """Stop the timer if it hasn't finished yet"""
        self.finished.set()

    def run(self):
        while not self.finished.is_set():
            self.function(*self.args, **self.kwargs)
            self.finished.wait(self.interval)

def upd_status(progress_info):
    
    #print('\r', end="")
    if not progress_info['f'] is None:
        description = f"f: {progress_info['f']:0.5g} | "
        description += f"mn: {progress_info['mn']:0.5f} | "
        description += f"mx: {-progress_info['mx']:0.5f} |"
        if not progress_info['f_prev'] is None:
            description += f"f prev: {progress_info['f_prev']:0.5g} | "
    else:
        description = ""
    for i in range(2 * M + 5):
        sys.stdout.write('\033[A')
        print('\r', ' '*200, '\r', end="")
    cur_time = time.time()
    t_delta = cur_time - progress_info['t_start']
    print(description, progress_info['it'], 'it', '[' f"{t_delta:0.2f}", 's', \
            f"{t_delta / max(1, progress_info['it']):0.2f}", 's/it', \
            'last upd:', f"{progress_info['t_last'] - progress_info['t_start']:0.2f}", \
            'from last upd:', f"{cur_time - progress_info['t_last']:0.2f}", 's'
            ']', flush=True)
    x = progress_info['x']
    if not x is None:
        strategy_center, strategy_border = prepare_matrix(x[:cnt], x[-cnt:], M)

        print('Strategy Center', flush=True)
        print(strategy_to_string(strategy_center, M), flush=True)
        
        print('Strategy Border', flush=True)
        print(strategy_to_string(strategy_border, M), flush=True)

    

def step_optimize(res, func, progress_info=None):
    if not progress_info is None:
        with lock_progress_info:
            
            progress_info['it'] += 1
            
            cur_time = time.time()
            progress_info['t_delta'] = cur_time - progress_info['t_last']
            progress_info['t_ema'] = (progress_info['t_ema'] + progress_info['t_delta']) / 2
            progress_info['t_last'] = cur_time
            
            f, mn, mx = func(res, need_all=True)
            if progress_info['f'] is None or progress_info['f'] > f:
                progress_info['f_prev'] = progress_info['f']
                progress_info['x'] = res
                progress_info['f'] = f
                progress_info['mn'] = mn
                progress_info['mx'] = mx
    '''
    if not progress is None:
        f, mn, mx = func(res, need_all=True)
        progress.set_description(f"{f:0.5f} | mn = {mn:0.5f} | mx = {mx:0.5f}")
        progress.update(1)
    '''

def max_objective(strategy_center_0, strategy_border, M):
    x0 = strategy_center_0
    #try:
    def max_func(x):
        return -objective_function_except(x, strategy_border, M)
    bounds = [bound for i in range(len(x0))]
    res = minimize(max_func, x0, bounds=bounds, method='Powell', tol=1e-10, options={'maxiter':10000, 'xtol': 1e-10, 'ftol': 1e-10})
    if np.isnan(res.fun):
        fun = max_func(res.x)
        if np.isnan(fun):
            print(fun, res.x, M)
        res.fun = fun
    if np.isnan(res.fun):
        print('Max BUG!!!!!!!')
        print(res.x, strategy_border)
    #, callback=partial(step_optimize, func=max_func, progress=progress_inner)
    #finally:
        #progress.close()
    return res.fun, res

def min_objective(strategy_center, strategy_border_0, M):
    x0 = strategy_border_0
    #progress = tqdm(mininterval=1)
    #try:
    def min_func(x):
        return objective_function_except(strategy_center, x, M)
    bounds = [bound for i in range(len(x0))]
    res = minimize(min_func, x0, bounds=bounds, method='Powell', tol=1e-10, options={'maxiter':10000, 'xtol': 1e-10, 'ftol': 1e-10})
    if np.isnan(res.fun):
        res.fun = min_func(res.x)
    if np.isnan(res.fun):
        print('Min BUG!!!!!!!')
        print(strategy_center, res.x)
    #, callback=partial(step_optimize, func=min_func, progress=progress)
    #finally:
    #    progress.close()
    return res.fun, res

def min_max_fun(x, need_all=False):
    strategy_center = x[:cnt]
    strategy_border = x[-cnt:]
    #mx, res1 = max_objective(strategy_border, M)
    #mn, res2 = min_objective(strategy_center, M)

    mx, res1 = random_search(max_objective, M, num_iter=1, strategy_center=None, strategy_border=strategy_border)
    mn, res2 = random_search(min_objective, M, num_iter=1, strategy_center=strategy_center, strategy_border=None)

    '''print()
    print('#' * 20)
    print(strategy_border)
    print(res1)
    print(strategy_center)
    print(res2)
    print(mx - mn, mn, mx)'''
    if need_all:
        return -mx - mn, mn, mx
    else:
        return -mx - mn

def min_dist(strategy_center_flat, strategy_border_flat, need_progress=True, progress_info=None):
    x0 = np.concatenate([strategy_center_flat, strategy_border_flat])
    
    if need_progress:
        progress = tqdm(mininterval=1)
    else:
        progress = None
    try:
        bounds = [bound for i in range(len(x0))]
        res = minimize(min_max_fun, x0, bounds=bounds, method='Powell', \
                        tol=1e-10, options={'maxiter':10000, 'xtol': 1e-10, 'ftol': 1e-10}, \
                        callback=partial(step_optimize, func=min_max_fun, progress_info=progress_info))
        if np.isnan(res.fun):
            res.fun = min_max_fun(res.x)
    finally:
        if not progress is None:
            progress.close()
    return res

def random_search(opt_func, M, num_iter=100, strategy_center=None, strategy_border=None):
    ans = None
    #print(strategy_center, strategy_border)
    for i in range(num_iter):
        if strategy_center is None:
            strategy_center_0 = np.random.uniform(low=bound[0], high=bound[1], size=cnt)
        else:
            strategy_center_0 = strategy_center
        if strategy_border is None:
            strategy_border_0 = np.random.uniform(low=bound[0], high=bound[1], size=cnt)
        else:
            strategy_border_0 = strategy_border
        #print(strategy_center_0, strategy_border_0)
        val, res = opt_func(strategy_center_0, strategy_border_0, M)
        #print(strategy_center_0, strategy_border_0, val)
        if ans is None:
            ans = res
        elif ans.fun > res.fun:
            ans = res
    return ans.fun, ans


def random_search_all(num_iter):
    ans = None
    init_ans = None
    for i in range(num_iter):
        strategy_center_flat = np.random.uniform(low=bound_init[0], high=bound_init[1], size=cnt)
        strategy_border_flat = np.random.uniform(low=bound_init[0], high=bound_init[1], size=cnt)
        init = (strategy_center_flat, strategy_border_flat)
        res = min_dist(strategy_center_flat, strategy_border_flat)
        if ans is None:
            ans = res
            init_ans = init
        elif ans.fun > res.fun:
            ans = res
            init_ans = init
    return ans, init_ans

from multiprocessing import Pool
def random_search_all_parallel(progress_info, num_iter=16, num_threads=16):
    args = []
    inits = []
    for i in range(num_iter):
        strategy_center_flat = np.random.uniform(low=bound_init[0], high=bound_init[1], size=cnt)
        strategy_border_flat = np.random.uniform(low=bound_init[0], high=bound_init[1], size=cnt)
        init = (strategy_center_flat, strategy_border_flat,)
        inits += [init]
        args += [(strategy_center_flat, strategy_border_flat, False, progress_info)]
    p = Pool(num_threads)
    res_all = p.starmap(min_dist, args)
    
    ans = None
    init_ans = None
    for i in range(num_iter):
        res = res_all[i]
        init = inits[i]
        if ans is None:
            ans = res
            init_ans = init
        elif ans.fun > res.fun:
            ans = res
            init_ans = init
    return ans, init_ans

def strategy_to_string(strategy, M):
    strategy_s = ''
    for i in range(M):
        strategy_t = []
        for j in range(M):
            strategy_t += [f'{strategy[i, j]:0.3f}']
        strategy_s += " ".join(strategy_t) + '\n'
    return strategy_s

if __name__ == '__main__':
    import multiprocessing
    manager = multiprocessing.Manager()
    progress_info_value = manager.dict(progress_info)
    status_updater = Timer(1, upd_status, [progress_info_value])
    status_updater.start()

    be = time.time()
    res, init_res = random_search_all_parallel(progress_info_value, num_iter=cnt * 4)
    f, mn, mx = min_max_fun(res.x, need_all=True)
    strategy_center, strategy_border = prepare_matrix(res.x[:cnt], res.x[-cnt:], M)
    en = time.time()
    status_updater.cancel()

    t_run = en - be
    
    with open('res.txt', 'a') as file:
        print('N:', N, 'M:', M, 'cnt:', cnt, 'bound:', bound, \
            'f:', f, 'mn:', mn, 'mx:', -mx, \
            'Time:', t_run, file=file)
        print('Strategy Center', file=file)
        print(strategy_to_string(strategy_center, M), file=file)
        
        print('Strategy Border', file=file)
        print(strategy_to_string(strategy_border, M), file=file)

        print('Initial x:', init_res, file=file)
        print('Final   x:', res.x, file=file)
        print()

        print(res, '\n', file=file)
    
    #max_objective(strategy_border, M)
    #min_objective(strategy_center, M)
    #res = minimize(lambda x: objective_function(x[0], x[1], M), x0, method='Nelder-Mead', tol=1e-6)