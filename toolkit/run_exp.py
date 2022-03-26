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
        'scp', file, f'{rem_host}:~/{rem_workspace}/'
    ]).returncode)


def exec_on_rem_workspace(rem_host, rem_workspace, cmds):
    cmds = [f'cd {rem_workspace}'] + cmds
    exit_if_error(subprocess.run([
        'ssh', rem_host, '; '.join(cmds)
    ]).returncode)


def prepare_workspace(rem_host: str, rem_workspace: str,
                      username: str, filename: str, filepath: str,
                      job: str, gpu: int, max_time: str,
                      job_file: str, output_dir: str, gtype: Union[None, str],
                      ckpt: Union[None,str], node: Union[None, str],
                      mem: Union[None, str]):
    # create workspace if not exists
    exit_if_error(subprocess.run([
        'ssh', rem_host, f'mkdir -p {rem_workspace}'
    ]).returncode)

    # make output dir and remove .nv folder
    exec_on_rem_workspace(rem_host=rem_host, rem_workspace=rem_workspace, cmds=[
        f'mkdir -p {output_dir}',
         'rm -rf ~/.nv/',
         f'cp {ckpt}/* {output_dir}' if ckpt else ':'
    ])

    # copy ginfile to remote
    send_to_server(filepath, rem_host, os.path.join(rem_workspace, output_dir))

    if username[0] == 'p':
        jobname = 'hourglass'
    elif username[0] == 's':
        jobname = 'policygrad'
    elif username[0] == 'd':
        jobname = 'python3'

    # prepare job
    job_str = R'''#!/bin/bash
rm -rf ~/.nv/
nvidia-smi -L
nvidia-smi

{job}

echo "Welcome to Vice City. Welcome to the 1980s."
    '''.format(
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

    send_to_server(job_file, rem_host, os.path.join(rem_workspace, output_dir))

    print('[INFO] Workspace prepared')


def create_job(branch: str,
               output_dir: str, gpu_count: int,
               ginfile: str) -> str:
    with open('neptune_props.json') as neptune_props_file:
      neptune_props = json.load(neptune_props_file)
    envs = [('TF_FORCE_GPU_ALLOW_GROWTH','true'),
            ('LD_LIBRARY_PATH','/usr/local/cuda-11/lib64:$LD_LIBRARY_PATH'),
            ('LD_LIBRARY_PATH','/usr/lib/cuda/lib64:$LD_LIBRARY_PATH'),
            ('NEPTUNE_API_TOKEN', 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlMTcxMzI2My1jOTY1LTQ5MjAtOGMzNC1jNmNhMzRlOGI3MGUifQ=='),
            ('TRAX_BRANCH', branch),
            ('NEPTUNE_PROJECT', neptune_props['NEPTUNE_PROJECT']),
            ('NEPTUNE_TOKEN', neptune_props['NEPTUNE_TOKEN']),
            ('XLA_PYTHON_CLIENT_PREALLOCATE', 'false'),
            ('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform'),
            ('EXPERIMENT_PATH', f'{_rem_host}:~/{_rem_workspace}/{output_dir}')]

    envs_bash = '\n'.join(
        f'export {k}={v}' for k,v in envs
    )

    job = '''
ulimit -n 60000
. /home/pnawrot/venv/bin/activate

{environment}
git clone https://github.com/Vatican-X-Formers/xl.git --branch {branch}
mv {ginfile} xl/
cd xl

rm -rf data
ln -s /home/pnawrot/piotrek/datasets/ data

C={ginfile} bash scripts/run_exp.sh {gpu_count}
    '''.format(
        branch=branch,
        output_dir=output_dir,
        environment=envs_bash,
        gpu_count=gpu_count,
        ginfile=ginfile,
    )

    print('[INFO] Job generated')

    return job


def run_job(rem_host: str, rem_workspace: str, job_file: str, jobid: int,
            out_file: str):
    cmds = [f'nohup srun --jobid={jobid} --output={out_file} bash {job_file} >/dev/null 2>/dev/null </dev/null &']
    exec_on_rem_workspace(rem_host=rem_host, rem_workspace=rem_workspace,
                          cmds=cmds)
    print('Job submitted')


def deploy_job(filepath: str, filename:str,
               username: str,
               branch: str, gpu:int, max_time: str,
               gtype: Union[str, None],
               ckpt: Union[str, None],
               node: Union[str, None],
               mem: Union[str, None],
               jobid: Union[int, None]) -> None:

    _date = time.strftime("%Y%m%d_%H%M%S")
    _out_file = _date+'_'+'.out'
    _job_file = 'jobtask.txt'

    _out_dir = filename+'_'+branch+'_'+_date

    job = create_job(branch=branch, output_dir=_out_dir,
                     gpu_count=gpu, ginfile=filename)
    prepare_workspace(rem_host=_rem_host, rem_workspace=_rem_workspace,
                      username=username, filepath=filepath,
                      filename=filename, job=job, gpu=gpu, max_time=max_time,
                      job_file= _job_file,
                      output_dir=_out_dir, gtype=gtype, ckpt=ckpt, node=node,
                      mem=mem)
    run_job(rem_host=_rem_host, rem_workspace=os.path.join(_rem_workspace, _out_dir),
            job_file=_job_file, jobid=jobid, out_file=_out_file)

    print(f'Output will be saved in\n{_rem_host}:~/{_rem_workspace}/{_out_dir}')
    os.remove(_job_file)


def target_info(filepath):
    filename = path_leaf(filepath)
    return filename, filepath


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
        '--ckpt', type=str, required=False, help='Folder name in vatican workspace where checkpoint is stored')
    parser.add_argument(
        '--jobid', type=str, required=False,)

    args = parser.parse_args()
    print(args.script, args.gin, bool(args.script), bool(args.gin))
    if not (bool(args.gin) ^ bool(args.script)):
        parser.error("One of --gin and --script required")

    _rem_host = f'{args.user}@entropy.mimuw.edu.pl'
    _rem_workspace = 'vatican_trax_workspace'

    gins = [os.path.join(args.gin, f) for f in os.listdir(args.gin)] if os.path.isdir(args.gin) else [args.gin]

    for gin in gins:
        filename, filepath = target_info(gin)
        time.sleep(2)
        deploy_job(username=args.user, branch=args.branch,
                gpu=args.gpu_count, max_time=args.time,
                filename=filename, filepath=filepath,
                gtype = args.gpu_type, ckpt=args.ckpt,
                node=args.node, mem=args.mem, jobid=args.jobid)
