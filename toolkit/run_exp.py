import argparse
import typing
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
                      job_file: str):
    # create workspace if not exists
    exit_if_error(subprocess.run([
        'ssh', rem_host, f'mkdir -p {rem_workspace}'
    ]).returncode)

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

{job}

echo "Welcome to Vice City. Welcome to the 1980s."
    '''.format(
        username=username,
        out_file=out_file,
        gpu=gpu,
        job=job
    )

    print('[DEBUG] Job configuration')
    print(job_str)
    with open(job_file, 'w') as output:
        output.write(job_str)
    
    # copy ginfile to remote
    send_to_server(job_file, rem_host, rem_workspace)
    
    print('[INFO] Workspace prepared') 

def create_job(ginfile: str, branch: str) -> str:
    
    job = '''
pip3 install matplotlib
pip3 install -q git+https://github.com/Vatican-X-Formers/trax.git@{branch}
pip3 install -q gin
python3 -m trax.trainer --config_file={ginfile}
    '''.format(
        branch=branch,
        ginfile=ginfile
    )

    print('[INFO] Job generated')

    return job
    
def run_job(rem_host: str, rem_workspace: str, job_file: str):
    cmds=[f'sbatch {job_file}']
    exec_on_rem_workspace(rem_host=rem_host, rem_workspace=rem_workspace,
                          cmds=cmds)

def deploy_model(ginfile: str, username: str, branch: str, gpu:int) -> None:

    _rem_host = f'{username}@entropy.mimuw.edu.pl'
    _rem_workspace = 'vatican_trax_workspace'
    _out_file = time.strftime("%Y%m%d_%H%M%S.out")
    _job_file = 'jobtask.txt'

    job = create_job(ginfile=ginfile, branch=branch)
    prepare_workspace(rem_host=_rem_host, rem_workspace=_rem_workspace, 
                      username=username, ginfile=ginfile,
                      job=job, gpu=gpu, out_file=_out_file,
                      job_file= _job_file)
    run_job(rem_host=_rem_host, rem_workspace=_rem_workspace,
            job_file=_job_file)
    

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
    args = parser.parse_args()
    deploy_model(ginfile=args.gin, username=args.user, branch=args.branch,
                 gpu=args.gpu_count)
    