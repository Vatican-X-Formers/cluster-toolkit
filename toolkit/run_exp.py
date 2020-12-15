import argparse
from typing import Union
import subprocess
import time

def exit_if_error(code):
    if code != 0:
        print('Something went wrong...')
        exit(1)

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
                      username: str, ginfile: str,
                      job: str, gpu: int, out_file: str,
                      job_file: str, custom_script: str,
                      output_dir: str):
    # create workspace if not exists
    exit_if_error(subprocess.run([
        'ssh', rem_host, f'mkdir -p {rem_workspace}'
    ]).returncode)

    # make output dir
    exec_on_rem_workspace(rem_host=rem_host, rem_workspace=rem_workspace, cmds=[
        f'mkdir -p {output_dir}'
    ])

    # copy ginfile to remote
    send_to_server(ginfile, rem_host, rem_workspace)

    # prepare job
    job_str = R'''#!/bin/bash
#
#SBATCH --job-name=job_zpp_vatican_{username}
#SBATCH --partition=common
#SBATCH --qos=8gpu3d
#SBATCH --gres=gpu:{gpu}
#SBATCH --output={out_file}

nvidia-smi -L
echo $(nvidia-smi -L) >> {meta_file}
cat {ginfile} >> {meta_file}

{job}

mv {meta_file} {output_dir}
mv {out_file} {output_dir}
mv {job_file} {output_dir}
mv {ginfile} {output_dir}
mv {custom_script} {output_dir}

rm -rf trax venv

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

    print('[DEBUG] Job configuration')
    print(job_str)
    with open(job_file, 'w') as output:
        output.write(job_str)
    
    # copy ginfile to remote
    send_to_server(job_file, rem_host, rem_workspace)

    # copy custom script to remote
    if custom_script:
        send_to_server(custom_script, rem_host, rem_workspace)

    print('[INFO] Workspace prepared') 

def create_job(ginfile: str, branch: str, custom_script: str,
               output_dir: str) -> str:
    
    job = '''
git clone -b {branch} https://github.com/Vatican-X-Formers/trax.git
python3 -m venv venv
source venv/bin/activate
# export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
pip3 install numpy==1.19.0
pip3 install -q matplotlib
pip3 install tensor2tensor
pip3 install -e trax
pip3 install -q gin
python3 {custom_script}
python3 -m trax.trainer --config_file={ginfile} --output_dir={output_dir}
    '''.format(
        branch=branch,
        ginfile=ginfile,
        custom_script=custom_script if custom_script else '--version',
        output_dir=output_dir
    )

    print('[INFO] Job generated')

    return job
    
def run_job(rem_host: str, rem_workspace: str, job_file: str):
    cmds=[f'sbatch {job_file}']
    exec_on_rem_workspace(rem_host=rem_host, rem_workspace=rem_workspace,
                          cmds=cmds)
    print('Job submitted')

def deploy_model(ginfile: str, username: str,
                 branch: str, gpu:int, custom_script: Union[str, None]) -> None:
    _date = time.strftime("%Y%m%d_%H%M%S")
    _out_file = _date+'.out'
    _out_dir = _date
    _rem_host = f'{username}@entropy.mimuw.edu.pl'
    _rem_workspace = 'vatican_trax_workspace'
    _job_file = 'jobtask.txt'

    job = create_job(ginfile=ginfile, branch=branch, custom_script=custom_script,
                     output_dir=_out_dir)
    prepare_workspace(rem_host=_rem_host, rem_workspace=_rem_workspace, 
                      username=username, ginfile=ginfile,
                      job=job, gpu=gpu, out_file=_out_file,
                      job_file= _job_file, custom_script=custom_script,
                      output_dir=_out_dir)
    run_job(rem_host=_rem_host, rem_workspace=_rem_workspace,
            job_file=_job_file)
    print(f'Output will be saved in\n{_rem_host}:~/{_rem_workspace}/{_out_dir}')

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
    args = parser.parse_args()
    deploy_model(ginfile=args.gin, username=args.user, branch=args.branch,
                 gpu=args.gpu_count, custom_script=args.script)
    