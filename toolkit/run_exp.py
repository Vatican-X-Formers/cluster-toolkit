import argparse
import typing
import subprocess
import time

def exit_if_error(code):
    if code != 0:
        print('Something went wrong...')
        exit(1)

def send_to_server(file, remote_host, workspace):
    exit_if_error(subprocess.run([
        'scp', file, f'{remote_host}:~/{workspace}/'
    ]).returncode)

def prepare_workspace(username: str, ginfile: str, job: str, gpu: int):
    _rem_host = f'{username}@entropy.mimuw.edu.pl'
    _rem_workspace = 'vatican_trax_workspace'
    
    # create workspace if not exists
    exit_if_error(subprocess.run([
        'ssh', _rem_host, f'mkdir -p {_rem_workspace}'
    ]).returncode)

    # copy ginfile to remote
    send_to_server(ginfile, _rem_host, _rem_workspace)

    # prepare job
    job_str = R'''
#!/bin/bash
#
#SBATCH --job-name=job_zpp_vatican_{username}
#SBATCH --partition=common
#SBATCH --qos=8gpu3d
#SBATCH --gres=gpu:{gpu}
#SBATCH --output={jobname}

nvidia-smi -L

{job}

echo "Welcome to Vice City. Welcome to the 1980s."
    '''.format(
        username=username,
        jobname=time.strftime("%Y%m%d_%H%M%S.out"),
        gpu=gpu,
        job=job
    )

    print('[DEBUG] Job configuration')
    print(job_str)
    TMP_JOB_LOC = '.jobtmplocal.txt'
    with open(TMP_JOB_LOC, 'w') as output:
        output.write(job_str)
    
    # copy ginfile to remote
    send_to_server(TMP_JOB_LOC, _rem_host, _rem_workspace)
    
    print('[INFO] Workspace prepared') 

def create_job(ginfile: str, branch: str) -> str:
    
    job = '''
pip install -q git+https://github.com/Vatican-X-Formers/trax.git@{branch}
pip install -q gin
python -m trax.trainer --config_file={ginfile}
    '''.format(
        branch=branch,
        ginfile=ginfile
    )

    print('[INFO] Job generated')

    return job
    
def run_job():
    pass

def deploy_model(ginfile: str, username: str, branch: str, gpu:int) -> None:
    job = create_job(ginfile=ginfile, branch=branch)
    prepare_workspace(username=username, ginfile=ginfile,
                      job=job, gpu=gpu)
    run_job()
    

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
    