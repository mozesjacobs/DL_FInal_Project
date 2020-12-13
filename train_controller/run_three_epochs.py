from gym import logger as gymlogger
gymlogger.set_level(40) #error only

import pyglet
print(pyglet.version)
from os.path import exists
import os
import torch
import cma
from models import Controller
from tqdm import tqdm
import numpy as np
import sys
from utils.misc import RolloutGenerator, ASIZE, RSIZE, LSIZE
from utils.misc import load_parameters
from utils.misc import flatten_parameters
from pyvirtualdisplay import Display
from collections import deque
import pickle


if __name__ == "__main__":
    obj_file_name = "cma/cma.obj"
    if not exists("cma"):
        os.mkdir("cma")
    if not exists("ctrl"):
        os.mkdir("ctrl")
    display = Display(visible=0, size=(1400, 900))
    display.start()
    time_limit = 1000
    device = "cuda"
    pop_size = 4
    n_samples = 4
    target_return = 950
    disp = True
    controller = Controller(LSIZE, RSIZE, ASIZE)
    parameters = controller.parameters()
    vae_weights_path = "weights/vae_original.pt"
    mdrnn_weights_path = "weights/mdrnn.pt"
    rg = RolloutGenerator("weights/vae_original.pt", "weights/mdrnn.pt", device, time_limit)
    p_queue = deque()
    r_queue = deque()
    
    def evaluate(solutions, results, rollouts=10):
        """ Give current controller evaluation.
        Evaluation is minus the cumulated reward averaged over rollout runs.
        :args solutions: CMA set of solutions
        :args results: corresponding results
        :args rollouts: number of rollouts
        :returns: minus averaged cumulated reward
        """
        index_min = np.argmin(results)
        best_guess = solutions[index_min]
        restimates = []
        
        for s_id in range(rollouts):
            p_queue.append((s_id, best_guess))
        
        while len(p_queue) != 0:
            id, best_guess = p_queue.popleft()
            r_queue.append((id, rg.rollout(best_guess)))
        
        print("Evaluating...")
        for _ in tqdm(range(rollouts)):
            restimates.append(r_queue.popleft()[1])
        
        return best_guess, np.mean(restimates), np.std(restimates)
    
    if not exists("ctrl/ctrl.pt") or not exists(obj_file_name):
        epoch = 0
        log_step = 3
        parameters = controller.parameters()
        es = cma.CMAEvolutionStrategy(flatten_parameters(parameters), 0.1,
                                  {'popsize': pop_size})
        cur_best = None
        end = log_step
    else:
        state = torch.load("ctrl/ctrl.pt")
        epoch = state['epoch'] + 1
        cur_best = - state['reward']
        controller.load_state_dict(state['state_dict'])
        s = open(obj_file_name, 'rb').read()
        es = pickle.loads(s)
        log_step = 3
        end = epoch + log_step
    
    while epoch < end:
        print(f"Epoch: {epoch}")
        if cur_best is not None and - cur_best > target_return:
            print("Already better than target, breaking...")
            break
    
        r_list = [0] * pop_size  # result list
        solutions = es.ask()
        print(f"Number of solutions: {len(solutions)}")
        # push parameters to queue
        for s_id, s in enumerate(solutions):
            for _ in range(n_samples):
                p_queue.append((s_id, s))
        print(f"Filled up p_queue of size {len(p_queue)}")
        while len(p_queue) != 0:
          id, sol = p_queue.popleft()
          r_queue.append((id, rg.rollout(sol)))
        print(f"Filled up r_queue of size {len(r_queue)}")
        # retrieve results
        if disp:
            pbar = tqdm(total=pop_size * n_samples)
        for _ in range(pop_size * n_samples):
            r_s_id, r = r_queue.popleft()
            r_list[r_s_id] += r / n_samples
            if disp:
                pbar.update(1)
        print("Filled up r_list")
        if disp:
            pbar.close()
    
        es.tell(solutions, r_list)
        es.disp()
    
        # evaluation and saving
        if epoch % log_step == log_step - 1:
            best_params, best, std_best = evaluate(solutions, r_list)
            print("Current evaluation: {}".format(best))
            if not cur_best or cur_best > best:
                cur_best = best
                print(f"New best is {cur_best}")
                print("Saving new best with value {}+-{}...".format(-cur_best, std_best))
                load_parameters(best_params, controller)
                torch.save(
                    {'epoch': epoch,
                     'reward': - cur_best,
                     'state_dict': controller.state_dict()},
                    "ctrl/ctrl.pt")
            else:
                state = torch.load("ctrl/ctrl.pt")
                torch.save(
                    { 'epoch': epoch,
                      'reward': state['reward'],
                      'state_dict': state['state_dict']}, "ctrl/ctrl.pt")
            
            s = es.pickle_dumps()
            open(obj_file_name, 'wb').write(s)
            if - best > target_return:
                print("Terminating controller training with value {}...".format(best))
                break
        epoch += 1
    display.stop()
