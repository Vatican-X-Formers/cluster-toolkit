import argparse
from typing import Union
import subprocess
import time
import os
import ntpath
import time
import string
import random
import json

def v1():
    return f"{''.join([random.choice(string.ascii_lowercase) for _ in range(2)])}-{random.choice(string.digits)}"

def v2():
    return f"{''.join([random.choice(string.ascii_lowercase) for _ in range(2)])}@Transformer"

def v3():
    return f"{random.randint(0,999)}"

def v4():
    return 'python3'

job_signature = {'y': v1, 'a':v2, 'j':v3, 'w':v4}

def gjn(sig):
    return job_signature[sig]()

def exit_if_error(code):
    if code != 0:
        print('Something went wrong...')
        exit(1)

def path_leaf(path):
    if path is None:
        return None
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def send_to_server(file, rem_host, rem_workspace):
    exit_if_error(subprocess.run([
        'scp', file, f'{rem_host}:~/{rem_workspace}/'
    ]).returncode)

def exec_on_rem_workspace(rem_host, rem_workspace, cmds):
    cmds = [f'cd {rem_workspace}'] + cmds
    exit_if_error(subprocess.run([
        'ssh', rem_host, '; '.join(cmds)
    ]).returncode)
    
def prepare_workspace(rem_host: str, rem_workspace: str,
                      username: str, filename: str, filepath: str,
                      job: str, gpu: int, max_time: str, out_file: str,
                      job_file: str, output_dir: str, gtype: Union[None, str],
                      ckpt: Union[None,str], node: Union[None, str],
                      mem: Union[None, str]):
    # create workspace if not exists
    exit_if_error(subprocess.run([
        'ssh', rem_host, f'mkdir -p {rem_workspace}'
    ]).returncode)

    # make output dir and remove .nv folder
    kth = username[2]
    exec_on_rem_workspace(rem_host=rem_host, rem_workspace=rem_workspace, cmds=[
        f'mkdir -p {output_dir}',
         'rm -rf ~/.nv/',
         f'cp {ckpt}/* {output_dir}' if ckpt else ':'
    ])



    # copy ginfile to remote
    send_to_server(filepath, rem_host, os.path.join(rem_workspace, output_dir))
    send_to_server('req.txt', rem_host, os.path.join(rem_workspace, output_dir))

    # prepare job
    job_str = R'''#!/bin/bash
#
#SBATCH --job-name={jobname}
#SBATCH --partition=common
#SBATCH --qos=16gpu3d
#SBATCH --gres=gpu:{gpu}
#SBATCH --time={time}
#SBATCH --output={out_file}
#SBATCH --mem=50G
{mem_requirement}
{nodelist}

# find / -type d -maxdepth 4 -name cuda 2>/dev/null
rm -rf ~/.nv/
nvidia-smi -L
nvidia-smi

echo $(nvidia-smi -L) >> {meta_file}
cat {dump_file} >> {meta_file}

{job}

# rm -rf trax venv

echo "Welcome to Vice City. Welcome to the 1980s."
    '''.format(
        jobname=gjn(kth),
        out_file=out_file,
        meta_file=out_file+'.meta',
        gpu=gpu if not gtype else f'{gtype}:{gpu}',
        time=max_time,
        job=job,
        dump_file=filename,
        output_dir=output_dir,
        nodelist=f"#SBATCH --nodelist={node}" if node else '',
        mem_requirement=f'#SBATCH --mem={mem}' if mem else '',
    )

    with open(job_file, 'w') as output:
        output.write(job_str)
    
    # copy ginfile to remote
    send_to_server(job_file, rem_host, os.path.join(rem_workspace, output_dir))

    print('[INFO] Workspace prepared') 


def create_job(exec_line: str, branch: str,
               output_dir: str, gpu_count: int,
               ginfile: str) -> str:
    with open('neptune_props.json') as neptune_props_file:
      neptune_props = json.load(neptune_props_file)
    envs = [('TF_FORCE_GPU_ALLOW_GROWTH','true'),
            ('LD_LIBRARY_PATH','/usr/local/cuda-11/lib64:$LD_LIBRARY_PATH'),
            ('LD_LIBRARY_PATH','/usr/lib/cuda/lib64:$LD_LIBRARY_PATH'),
            #('CUDA_HOME', '/usr/local/cuda-11'),
            #('PATH', '/usr/local/cuda-11/bin:/usr/local/cuda-11/lib64:$PATH'),
            ('NEPTUNE_API_TOKEN', 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlMTcxMzI2My1jOTY1LTQ5MjAtOGMzNC1jNmNhMzRlOGI3MGUifQ=='),
            ('TRAX_BRANCH', branch),
            ('NEPTUNE_PROJECT', neptune_props['NEPTUNE_PROJECT']),
            ('NEPTUNE_TOKEN', neptune_props['NEPTUNE_TOKEN']),
            ('XLA_PYTHON_CLIENT_PREALLOCATE', 'false'),
            ('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')]

    envs_bash = '\n'.join(
        f'export {k}={v}' for k,v in envs
    )

    job = '''
ulimit -n 60000
. ~/venv/bin/activate

{environment}
git clone https://github.com/Vatican-X-Formers/xl.git --branch {branch}
mv {ginfile} xl/pytorch
cd xl

ln -s ~/xl_ds_cache data
bash getdata.sh

cd pytorch
bash train.sh {ginfile} {gpu_count} --config gpu_{gpu_count}
    '''.format(
        branch=branch,
        exec_line=exec_line,
        output_dir=output_dir,
        environment=envs_bash,
        gpu_count=gpu_count,
        ginfile=ginfile,
    )

    print('[INFO] Job generated')

    return job
    
def run_job(rem_host: str, rem_workspace: str, job_file: str):
    cmds=[f'sbatch {job_file}']
    exec_on_rem_workspace(rem_host=rem_host, rem_workspace=rem_workspace,
                          cmds=cmds)
    print('Job submitted')

def deploy_job(filepath: str, filename:str, 
               exec_line, username: str,
               branch: str, gpu:int, max_time: str,
               gtype: Union[str, None],
               ckpt: Union[str, None],
               node: Union[str, None],
               mem: Union[str, None]) -> None:
    
    _date = time.strftime("%Y%m%d_%H%M%S")
    _out_file = _date+'_'+'.out'
    _job_file = 'jobtask.txt'


    _out_dir = filename+'_'+branch+'_'+_date

    job = create_job(exec_line=exec_line, branch=branch, output_dir=_out_dir,
                     gpu_count=gpu, ginfile=filename)
    prepare_workspace(rem_host=_rem_host, rem_workspace=_rem_workspace, 
                      username=username, filepath=filepath,
                      filename=filename, job=job, gpu=gpu, max_time=max_time,
                      out_file=_out_file, job_file= _job_file,
                      output_dir=_out_dir, gtype=gtype, ckpt=ckpt, node=node,
                      mem=mem)
    run_job(rem_host=_rem_host, rem_workspace=os.path.join(_rem_workspace, _out_dir),
            job_file=_job_file)

    print(f'Output will be saved in\n{_rem_host}:~/{_rem_workspace}/{_out_dir}')

def download_datasets(rem_host: str, rem_workspace: str):
    exec_on_rem_workspace(rem_host=rem_host, rem_workspace=rem_workspace, cmds=[
        'mkdir tensorflow_datasets',
        'cd tensorflow_datasets',
        'mkdir download',
        'wget http://image-net.org/small/valid_32x32.tar',
        'wget http://image-net.org/small/train_32x32.tar',
        'tar -C ~/tensorflow_datasets/download/ -xf valid_32x32.tar',
        'rm -rf valid_32x32.tar',
        'tar -C ~/tensorflow_datasets/download/ -xf train_32x32.tar',
        'rm -rf train_32x32.tar'
    ])

def install(user: str, rem_host: str, rem_workspace: str):
    reinstall(user=user, rem_host=rem_host, rem_workspace=rem_workspace)
    download_datasets(rem_host=rem_host, rem_workspace=rem_workspace)

def reinstall(user: str, rem_host: str, rem_workspace: str):

    with open('vatican.pth', 'w+') as vatican:
        vatican.write(f'/home/{user}/venv/lib/python3.8/site-packages')

    send_to_server(file='req.txt', rem_host=rem_host, rem_workspace=rem_workspace)
    send_to_server(file='vatican.pth', rem_host=rem_host, rem_workspace=rem_workspace)
    exec_on_rem_workspace(rem_host=rem_host, rem_workspace=rem_workspace, cmds=[
        'rm -rf venv',
        'python3 -m venv venv',
        'source venv/bin/activate',
        'pip3 install pip==20.2.4',
        'pip3 install cmake scikit-build',
        'pip3 install --upgrade pip setuptools wheel',
        'XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda pip3 install -r req.txt',
        'pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html',
        'pip3 install nvidia-pyindex git+https://github.com/NVIDIA/dllogger.git',
        'pip install pytorch-transformers==1.1.0 sacremoses==0.0.35 pynvml==8.0.4',
        'XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda pip3 install --upgrade jaxlib==0.1.57+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html',
        'XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda pip3 install git+https://github.com/syzymon/tensor2tensor.git@master',
        'deactivate'
    ])



def target_info(is_gin: bool, filepath):
    filename = path_leaf(filepath)
    if not is_gin:
        exec_line = f'XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda python3 {filename}'
    else:
        assert(gin)
        exec_line = f'XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda python3 -m trax.trainer --config_file={filename} --output_dir=./'
    return exec_line, filename, filepath



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gin', help='gin file path', required=False, type=str)
    parser.add_argument(
        '--user', help='username on the cluster', required=True, type=str)
    parser.add_argument(
        '--branch', help='branch name', required=True, type=str)
    parser.add_argument(
        '--gpu-count', help='number of gpu', default=1, type=int)
    parser.add_argument(
        '--time', help='maximum job time, see recent mail', default="3-0",
        type=str)
    parser.add_argument(
        '--gpu-type', help='type of gpu', type=str, choices=['1080ti', 'titanx', 'titanv', 'rtx2080ti'])
    parser.add_argument(
        '--node', help='type of gpu', required=False, type=str, choices=
        [f'asusgpu{i}' for i in range(1,7)]+['arnold', 'steven', 'sylvester', 'bruce'])
    parser.add_argument(
        '--mem', help='cpu memory', required=False, default=None, type=str)
    parser.add_argument(
        '--script', help='custom script', type=str, required=False)
    parser.add_argument(
        '--install', action='store_true', help='Install full global venv along with downloading dataset')
    parser.add_argument(
        '--reinstall', action='store_true', help='Reinstall full global venv - without readownloading dataset')
    parser.add_argument(
        '--ckpt', type=str, required=False, help='Folder name in vatican workspace where checkpoint is stored')


    args = parser.parse_args()
    print(args.script, args.gin, bool(args.script), bool(args.gin))
    if not (bool(args.gin) ^ bool(args.script)):
        parser.error("One of --gin and --script required")

    _rem_host = f'{args.user}@entropy.mimuw.edu.pl'
    _rem_workspace = 'vatican_trax_workspace'
 
    if args.install:
        install(user=args.user, rem_host=_rem_host, rem_workspace='')
    elif args.reinstall:
        reinstall(user=args.user, rem_host=_rem_host, rem_workspace='')

    if args.gin:
        gins = [os.path.join(args.gin, f) for f in os.listdir(args.gin)] if os.path.isdir(args.gin) else [args.gin]

        for gin in gins:

            exec_line, filename, filepath = target_info(True, gin)  
            time.sleep(2)
            deploy_job(username=args.user, branch=args.branch,
                    gpu=args.gpu_count, max_time=args.time,
                    filename=filename, filepath=filepath,
                    exec_line=exec_line, gtype = args.gpu_type, ckpt=args.ckpt,
                    node=args.node, mem=args.mem)
    else:
        exec_line, filename, filepath = target_info(False, args.script)  
        deploy_job(username=args.user, branch=args.branch,
                    gpu=args.gpu_count, max_time=args.time,
                    filename=filename, filepath=filepath,
                    exec_line=exec_line, gtype = args.gpu_type, ckpt=args.ckpt,
                    node=args.node, mem=args.mem)
        
