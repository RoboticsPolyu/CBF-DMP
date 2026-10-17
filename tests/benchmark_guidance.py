"""Compare a saved pre-optimization source with the current implementation locally.
Run from repository root: python tests/benchmark_guidance.py
"""
import ast
import contextlib
import io
import json
import math
from pathlib import Path
import statistics
import time
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'Config', 'PositionalEncoding', 'ObstacleEncoder', 'AttentionObstacleEncoder',
         'ConditionEmbedding', 'ObstacleAwareDiffusionTransformer', 'ObstacleAwareDiffusionProcess',
         'AeroDM', 'compute_barrier_and_grad', 'compute_barrier_and_grad_logistic',
         'pack_guidance_obstacles', '_batched_barrier'}
def load(path):
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in NAMES]
    ns = dict(torch=torch, nn=nn, F=F, math=math, time=time, MANEUVER_STYLES=range(23))
    exec(compile(tree, str(path), 'exec'), ns)
    return ns
old = load(ROOT/'output/guidance_benchmark/before.py')
new = load(ROOT/'AeroDM_ObstacleAware_Training_Test.py')
# Ensure configuration and model architecture have not changed.
for name in ('Config', 'PositionalEncoding', 'ConditionEmbedding', 'ObstacleAwareDiffusionTransformer'):
    trees = [ast.parse(p.read_text()) for p in (ROOT/'output/guidance_benchmark/before.py', ROOT/'AeroDM_ObstacleAware_Training_Test.py')]
    assert ast.dump(next(n for n in trees[0].body if getattr(n,'name',None)==name)) == ast.dump(next(n for n in trees[1].body if getattr(n,'name',None)==name))
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
def sync():
    if device.type == 'mps': torch.mps.synchronize()
def seed(n):
    torch.manual_seed(n)
def timed(fn, reps=5):
    for _ in range(2): fn()
    sync(); ts=[]
    for _ in range(reps):
        sync(); start=time.perf_counter(); fn(); sync(); ts.append(time.perf_counter()-start)
    return statistics.median(ts)
report={'device':str(device),'torch':torch.__version__,'barrier':[]}
seed(123)
x=torch.randn(32,60,10,device=device)
mean=torch.zeros(1,1,11,device=device); std=torch.ones_like(mean)*2
obs=[[{'center':torch.randn(3,device=device),'radius':0.5+j*0.1} for j in range(b%6)] for b in range(32)]
# Include inside, boundary and center cases; uneven and empty obstacle lists.
obs[0]=[{'center':torch.zeros(3,device=device),'radius':1.0}]
x[0,:3,1:4]=torch.tensor([[0,0,0],[0.5,0,0],[0.50001,0,0]],device=device)
for name in ('compute_barrier_and_grad','compute_barrier_and_grad_logistic'):
    kwargs={'sigma':new['Config'].barrier_sigma} if 'logistic' in name else {}
    for case in (None, [], [[] for _ in range(32)], obs[:7], obs):
        a=old[name](x,mean,std,case,**kwargs); b=new[name](x,mean,std,case,**kwargs)
        for av,bv in zip(a,b): torch.testing.assert_close(av,bv,atol=2e-5,rtol=2e-5)
    packed=new['pack_guidance_obstacles'](obs,x)
    a=old[name](x,mean,std,obs,**kwargs); b=new[name](x,mean,std,packed,**kwargs)
    entry={'name':name,'value_max_error':(a[0]-b[0]).abs().item(),'gradient_max_error':(a[1]-b[1]).abs().max().item()}
    entry['before_s']=timed(lambda:old[name](x,mean,std,obs,**kwargs))
    entry['after_s']=timed(lambda:new[name](x,mean,std,packed,**kwargs))
    report['barrier'].append(entry)
print(json.dumps(report),flush=True)
# Load the configured checkpoint, using its normalization for full sampling.
Config=new['Config']  # legacy checkpoint pickle refers to __main__.Config
cfg=Config()
checkpoint=torch.load(ROOT/cfg.model_save_dir/cfg.model_filename,map_location='cpu',weights_only=False)
a=old['AeroDM'](old['Config']()).to(device).eval()
b=new['AeroDM'](cfg).to(device).eval()
a.load_state_dict(checkpoint['model_state_dict']); b.load_state_dict(checkpoint['model_state_dict'])
mean=checkpoint['mean'].to(device); std=checkpoint['std'].to(device)
for model in (a,b): model.set_normalization_params(mean,std); model.set_obstacles_data(obs)
seed(55)
history=torch.randn(32,cfg.history_len,cfg.state_dim,device=device)*0.2
target=torch.randn(32,cfg.target_dim,device=device)*0.2
action=F.one_hot(torch.arange(32,device=device)%cfg.action_dim,cfg.action_dim).float()
def sample(model):
    seed(77)
    with contextlib.redirect_stdout(io.StringIO()):
        return model.sample(target,action,history,batch_size=32,enable_guidance=True,guidance_gamma=cfg.guidance_gamma)
report['sampling']=[]
for barrier in ('standard','logistic'):
    a.config.barrier_type=barrier; b.config.barrier_type=barrier
    before=sample(a); after=sample(b)
    diff=(before-after).abs()
    entry={'barrier':barrier,'max_abs_error':diff.max().item(),'mean_abs_error':diff.mean().item(),'finite':bool(torch.isfinite(after).all())}
    torch.testing.assert_close(before,after,atol=2e-4,rtol=2e-4)
    entry['before_s']=timed(lambda:sample(a),3)
    entry['after_s']=timed(lambda:sample(b),3)
    report['sampling'].append(entry)
    print(json.dumps(entry),flush=True)
report['checkpoint']=cfg.model_filename
report['batch_size']=32
report['steps']=cfg.diffusion_steps
(ROOT/'output/guidance_benchmark/results.json').write_text(json.dumps(report,indent=2))
print('PASS: configuration, barrier and full 60-step sampling checks',flush=True)
