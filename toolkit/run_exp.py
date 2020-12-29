import argparse
from typing import Union
import subprocess
import time
import os
import ntpath
import time


def exit_if_error(code):
    if code != 0:
        print('Something went wrong...')
        exit(1)

def path_leaf(path):
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
                      username: str, ginfile: str, ginpath: str,
                      job: str, gpu: int, out_file: str,
                      job_file: str, custom_script: str,
                      output_dir: str):
    # create workspace if not exists
    exit_if_error(subprocess.run([
        'ssh', rem_host, f'mkdir -p {rem_workspace}'
    ]).returncode)

    # make output dir and remove .nv folder
    exec_on_rem_workspace(rem_host=rem_host, rem_workspace=rem_workspace, cmds=[
        f'mkdir -p {output_dir}', 'rm -rf ~/.nv/'
    ])



    # copy ginfile to remote
    send_to_server(ginpath, rem_host, os.path.join(rem_workspace, output_dir))
    send_to_server('req.txt', rem_host, os.path.join(rem_workspace, output_dir))

    # prepare job
    job_str = R'''#!/bin/bash
#
#SBATCH --job-name=job_zpp_vatican_{username}
#SBATCH --partition=common
#SBATCH --qos=8gpu3d
#SBATCH --gres=gpu:{gpu}
#SBATCH --output={out_file}

# find / -type d -maxdepth 4 -name cuda 2>/dev/null
rm -rf ~/.nv/
nvidia-smi -L

echo $(nvidia-smi -L) >> {meta_file}
cat {ginfile} >> {meta_file}

{job}

# rm -rf trax venv

echo "Welcome to Vice City. Welcome to the 1980s."
    '''.format(
        username=username,
        out_file=out_file,
        meta_file=out_file+'.meta',
        gpu=gpu,
        job=job,
        ginfile=ginfile,
        job_file=job_file,
        custom_script=custom_script,
        output_dir=output_dir
    )

    with open(job_file, 'w') as output:
        output.write(job_str)
    
    # copy ginfile to remote
    send_to_server(job_file, rem_host, os.path.join(rem_workspace, output_dir))

    # copy custom script to remote
    if custom_script:
        send_to_server(custom_script, rem_host, os.path.join(rem_workspace, output_dir))

    print('[INFO] Workspace prepared') 


def create_job(ginfile: str, branch: str, custom_script: str,
               output_dir: str) -> str:
    envs = [('TF_FORCE_GPU_ALLOW_GROWTH','true'),
            ('LD_LIBRARY_PATH','/usr/local/cuda-11/lib64:$LD_LIBRARY_PATH'),
            ('LD_LIBRARY_PATH','/usr/lib/cuda/lib64:$LD_LIBRARY_PATH')]

    envs_bash = '\n'.join(
        f'export {k}={v}' for k,v in envs
    )

    job = '''
python3 -m venv venv
cp ../../vatican.pth venv/lib/python3.8/site-packages/
source venv/bin/activate
XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda pip3 install matplotlib wheel
XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda pip3 install --no-deps git+https://github.com/Vatican-X-Formers/trax.git@{branch}

{environment}

XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda python3 {custom_script}
XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda python3 -m trax.trainer --config_file={ginfile} --output_dir=./
mv eval/* ./
mv train/* ./
rm -rf eval train venv
    '''.format(
        branch=branch,
        ginfile=ginfile,
        custom_script=custom_script if custom_script else '--version',
        output_dir=output_dir,
        environment=envs_bash
    )

    print('[INFO] Job generated')

    return job
    
def run_job(rem_host: str, rem_workspace: str, job_file: str):
    cmds=[f'sbatch {job_file}']
    exec_on_rem_workspace(rem_host=rem_host, rem_workspace=rem_workspace,
                          cmds=cmds)
    print('Job submitted')

def deploy_job(ginpath: str, username: str,
                 branch: str, gpu:int, custom_script: Union[str, None]) -> None:
    
    _date = time.strftime("%Y%m%d_%H%M%S")
    _out_file = _date+'_'+'.out'
    _job_file = 'jobtask.txt'

    # overwrite ginpath with ginfile name
    ginfile = path_leaf(ginpath)
    _out_dir = ginfile+'_'+branch+'_'+_date

    job = create_job(ginfile=ginfile, branch=branch, custom_script=custom_script,
                     output_dir=_out_dir)
    prepare_workspace(rem_host=_rem_host, rem_workspace=_rem_workspace, 
                      username=username, ginfile=ginfile, ginpath=ginpath,
                      job=job, gpu=gpu, out_file=_out_file,
                      job_file= _job_file, custom_script=custom_script,
                      output_dir=_out_dir)
    run_job(rem_host=_rem_host, rem_workspace=os.path.join(_rem_workspace, _out_dir),
            job_file=_job_file)

    print(f'Output will be saved in\n{_rem_host}:~/{_rem_workspace}/{_out_dir}')

def install(user: str, rem_host: str, rem_workspace: str):

    with open('vatican.pth', 'w+') as vatican:
        vatican.write(f'/home/{user}/venv/lib/python3.8/site-packages')

    send_to_server(file='req.txt', rem_host=rem_host, rem_workspace=rem_workspace)
    send_to_server(file='vatican.pth', rem_host=rem_host, rem_workspace=rem_workspace)
    exec_on_rem_workspace(rem_host=rem_host, rem_workspace=rem_workspace, cmds=[
        'rm -rf venv',
        'python3 -m venv venv',
        'source venv/bin/activate',
        'pip3 install --upgrade pip',
        'pip3 install cmake scikit-build',
        'pip3 install --upgrade pip setuptools wheel',
        'XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda pip3 install -r req.txt',
        'XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda pip3 install --upgrade jax jaxlib==0.1.57+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html',
        'XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda pip3 install git+https://github.com/Vatican-X-Formers/tensor2tensor.git@imagenet_funnel',
        'deactivate'
    ])  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gin', help='gin file path', required=True, type=str)
    parser.add_argument(
        '--user', help='username on the cluster', required=True, type=str)
    parser.add_argument(
        '--branch', help='branch name', required=True, type=str)
    parser.add_argument(
        '--gpu-count', help='number of gpu', required=False, default=1, type=int)
    parser.add_argument(
        '--script', help='custom script', required=False, type=str)
    parser.add_argument(
        '--install', action='store_true')

    args = parser.parse_args()

    gins = [os.path.join(args.gin, f) for f in os.listdir(args.gin)] if os.path.isdir(args.gin) else [args.gin]

    _rem_host = f'{args.user}@entropy.mimuw.edu.pl'
    _rem_workspace = 'vatican_trax_workspace'
 

    if args.install:
        install(user=args.user, rem_host=_rem_host, rem_workspace='')

    for gin in gins:
        time.sleep(2)
        deploy_job(ginpath=gin, username=args.user, branch=args.branch,
                     gpu=args.gpu_count, custom_script=args.script)
        