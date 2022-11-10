import argparse
from typing import Union
import subprocess
import time
import os
import ntpath
import json


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
        'scp', file, f'{rem_host}:{rem_workspace}/'
    ]).returncode)


def exec_on_rem_workspace(rem_host, rem_workspace, cmds):
    cmds = [f'cd {rem_workspace}'] + cmds
    exit_if_error(subprocess.run([
        'ssh', rem_host, '; '.join(cmds)
    ]).returncode)


def prepare_workspace(rem_host: str, rem_workspace: str,
                      filename: str, filepath: str,
                      job: str, gpu: int, max_time: str,
                      job_file: str, output_dir: str, gtype: Union[None, str],
                      node: Union[None, str],
                      mem: Union[None, str]):
    # create workspace if not exists
    exit_if_error(subprocess.run([
        'ssh', rem_host, f'mkdir -p {rem_workspace}'
    ]).returncode)

    # make output dir
    exec_on_rem_workspace(rem_host=rem_host, rem_workspace=rem_workspace, cmds=[
        f'mkdir -m 777 -p {output_dir}',
    ])

    # copy ginfile to remote
    send_to_server(filepath, rem_host, os.path.join(rem_workspace, output_dir))

    # prepare job
    job_str = R'''#!/bin/bash
#SBATCH --gres=gpu:{gpu}
#SBATCH --time={max_time}
#SBATCH -A NLP-CDT-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

source ~/.bashrc
nvidia-smi -L
nvidia-smi

{job}
'''.format(
        job=job,
        gpu=gpu,
        max_time=max_time
    )

    with open(job_file, 'w') as output:
        output.write(job_str)

    send_to_server(job_file, rem_host, os.path.join(rem_workspace, output_dir))
    print('[INFO] Workspace prepared')


def create_job(branch: str,
               output_dir: str,
               gpu_count: int,
               ginfile: str) -> str:
    with open('neptune_props.json') as neptune_props_file:
        neptune_props = json.load(neptune_props_file)

    envs = [
        ('NEPTUNE_API_TOKEN', 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlMTcxMzI2My1jOTY1LTQ5MjAtOGMzNC1jNmNhMzRlOGI3MGUifQ=='),
        ('TRAX_BRANCH', branch),
        ('NEPTUNE_PROJECT', neptune_props['NEPTUNE_PROJECT']),
        ('NEPTUNE_TOKEN', neptune_props['NEPTUNE_TOKEN']),
        ('EXPERIMENT_PATH', f'{_rem_host}:{_rem_workspace}/{output_dir}')
    ]

    envs_bash = '\n'.join(
        f'export {k}={v}' for k, v in envs
    )

    job = '''
conda activate cos

{environment}

git clone https://github.com/PiotrNawrot/hourglass --branch {branch} xl
mv {ginfile} xl/
cd xl
ln -s ~/data/ data

C={ginfile} bash scripts/run_exp.sh {gpu_count}
    '''.format(
        branch=branch,
        environment=envs_bash,
        gpu_count=gpu_count,
        ginfile=ginfile,
    )

    print('[INFO] Job generated')

    return job


def run_job(rem_host: str, rem_workspace: str, job_file: str, out_file: str):
    cmds = [f'sbatch {job_file}']
    exec_on_rem_workspace(rem_host=rem_host, rem_workspace=rem_workspace, cmds=cmds)
    print('Job submitted')


def deploy_job(filepath: str, filename:str,
               branch: str, gpu:int, max_time: str,
               gtype: Union[str, None],
               node: Union[str, None],
               mem: Union[str, None],) -> None:

    _date = time.strftime("%Y%m%d_%H%M%S")
    _out_file = _date+'_'+'.out'
    _job_file = 'jobtask.txt'

    _out_dir = filename+'_'+branch+'_'+_date

    job = create_job(branch=branch, output_dir=_out_dir,
                     gpu_count=gpu, ginfile=filename)

    prepare_workspace(rem_host=_rem_host, rem_workspace=_rem_workspace,
                      filepath=filepath,
                      filename=filename, job=job, gpu=gpu, max_time=max_time,
                      job_file= _job_file,
                      output_dir=_out_dir, gtype=gtype, node=node,
                      mem=mem)

    run_job(rem_host=_rem_host, rem_workspace=os.path.join(_rem_workspace, _out_dir),
            job_file=_job_file, out_file=_out_file)

    print(f'Output will be saved in\n{_rem_host}:{_rem_workspace}/{_out_dir}')
    os.remove(_job_file)


def target_info(filepath):
    filename = path_leaf(filepath)
    return filename, filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gin', help='gin file path', required=False, type=str)
    parser.add_argument(
        '--branch', help='branch name', required=True, type=str)
    parser.add_argument(
        '--gpu-count', help='number of gpu', default=1, type=int)
    parser.add_argument(
        '--time', help='maximum job time, see recent mail', default="1-0",
        type=str)
    parser.add_argument(
        '--gpu-type', help='type of gpu', type=str, choices=['1080ti', 'titanx', 'titanv', 'rtx2080ti'])
    parser.add_argument(
        '--node', help='type of gpu', required=False, type=str, choices=
        [f'asusgpu{i}' for i in range(1,7)]+['arnold', 'steven', 'sylvester', 'bruce'])
    parser.add_argument(
        '--mem', help='cpu memory', required=False, default=None, type=str)

    args = parser.parse_args()
    print(args.gin, bool(args.gin))

    _rem_host = f'csd3'
    _rem_workspace = '~/experiments'

    gins = [os.path.join(args.gin, f) for f in os.listdir(args.gin)] if os.path.isdir(args.gin) else [args.gin]

    for gin in gins:
        filename, filepath = target_info(gin)
        time.sleep(2)
        deploy_job(branch=args.branch,
                   gpu=args.gpu_count, max_time=args.time,
                   filename=filename, filepath=filepath,
                   gtype=args.gpu_type,
                   node=args.node, mem=args.mem)
