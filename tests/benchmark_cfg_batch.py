"""Mac MPS correctness and latency comparison for fused CFG; 60 steps."""
import contextlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));os.environ.setdefault('MPLBACKEND','Agg')
import AeroDM_ObstacleAware_Training_Test as current
from AeroDM_ObstacleAware_Training_Test import Config
spec=importlib.util.spec_from_file_location('cfg_before',ROOT/'output/cfg_batch_benchmark/before.py')
before=importlib.util.module_from_spec(spec);spec.loader.exec_module(before)
assert torch.backends.mps.is_available(), 'MPS is required'
device=torch.device('mps')
checkpoint=current.load_model_checkpoint(ROOT/Config.model_save_dir/Config.model_filename,'cpu')
models=[before.AeroDM(before.Config()).to(device).eval(),current.AeroDM(Config()).to(device).eval()]
for model in models:
    model.load_state_dict(checkpoint['model_state_dict'])
    model.set_normalization_params(checkpoint['mean'].to(device),checkpoint['std'].to(device))
scenes=np.load(ROOT/'output/rk3588_20261017/scenes.npz')
report={'device':'mps','torch':torch.__version__,'checkpoint':Config.model_filename,'steps':60,'warmup':2,'repeats':3,'results':[]}
quiet=open(os.devnull,'w')
def make_batch(start,count):
    h,t,a=[torch.from_numpy(scenes[k][start:start+count]).to(device) for k in ('history','target','action')]
    obs=[[dict(center=torch.from_numpy(scenes['centers'][i,j]).to(device),radius=float(scenes['radii'][i,j])) for j in range(int(scenes['counts'][i]))] for i in range(start,start+count)]
    return h,t,a,obs

def sample(model,batch,seed):
    h,t,a,obs=batch;model.obstacles_data=obs;torch.manual_seed(seed)
    torch.mps.synchronize();start=time.perf_counter()
    with contextlib.redirect_stdout(quiet),torch.no_grad():
        value=model.sample(t,a,h,batch_size=h.shape[0],enable_guidance=True,guidance_gamma=model.config.guidance_gamma)
    torch.mps.synchronize();return time.perf_counter()-start,value

# One-step branch checks: final/intermediate step, unguided, scale=1,
# shared history, missing optional inputs, and obstacle encoding enabled.
for encoder in (False,True):
    for m in models:m.config.enable_obstacle_encoding=encoder
    for mode in ('cfg','unguided','scale1','shared_history','optional_none'):
        h,target,action,obs=make_batch(0,3)
        if mode=='shared_history':h=h[:1]
        if mode=='optional_none':target=None;h=None
        for m in models:m.config.guidance_scale=1.0 if mode=='scale1' else 2.0
        for step in (0,29,59):
            torch.manual_seed(123);x=torch.randn(3,60,10,device=device);t=torch.full((3,),step,device=device,dtype=torch.long)
            vals=[];calls=[]
            for m in models:
                count=[0]
                hook=m.diffusion_model.register_forward_hook(lambda *args,c=count:c.__setitem__(0,c[0]+1))
                torch.manual_seed(456)
                vals.append(m.diffusion_process.p_sample(m.diffusion_model,x,t,target,action,h,enable_guidance=mode!='unguided',guidance_gamma=m.config.guidance_gamma,mean=m.mean,std=m.std,obstacles_data=obs,timestep=step))
                hook.remove();calls.append(count[0])
            expected=[2,1] if mode not in ('unguided','scale1') else [1,1]
            assert calls==expected,(mode,calls)
            torch.testing.assert_close(vals[0],vals[1],atol=2e-4,rtol=2e-4)
for m in models:
    m.config.enable_obstacle_encoding=Config.enable_obstacle_encoding
    m.config.guidance_scale=Config.guidance_scale
print('PASS one-step branch, encoder and forward-call checks',flush=True)
for batch_size in (1,32):
    batches=[make_batch(i,min(batch_size,32-i)) for i in range(0,32,batch_size)]
    for m in models:
        for i in range(2):sample(m,batches[0],90+i)
    durations=[[],[]];max_error=0.;diff_sum=0.;num_values=0
    for repeat in range(3):
        round_times=[0.,0.]
        for i,batch in enumerate(batches):
            values={}
            for idx in ((0,1) if (repeat+i)%2==0 else (1,0)):
                elapsed,value=sample(models[idx],batch,1000+repeat*len(batches)+i)
                round_times[idx]+=elapsed;values[idx]=value
            torch.testing.assert_close(values[0],values[1],atol=2e-4,rtol=2e-4)
            diff=(values[0]-values[1]).abs();max_error=max(max_error,diff.max().item());diff_sum+=diff.sum().item();num_values+=diff.numel()
        for idx in (0,1):durations[idx].append(round_times[idx])
        print('batch',batch_size,'round',repeat+1,'total_seconds',round_times,flush=True)
    row={'batch_size':batch_size,'trajectories_per_round':32,'max_abs_error':max_error,'mean_abs_error':diff_sum/num_values,
         'before_total_s':float(np.mean(durations[0])),'after_total_s':float(np.mean(durations[1])),
         'before_batch_ms':float(np.mean(durations[0]))/len(batches)*1000,'after_batch_ms':float(np.mean(durations[1]))/len(batches)*1000,
         'repeat_totals_s':durations}
    report['results'].append(row)
    print(row,flush=True)
(ROOT/'output/cfg_batch_benchmark/results.json').write_text(json.dumps(report,indent=2))
print('PASS full-sampling comparison',flush=True)
quiet.close()
