# Cluster Toolkit
Toolkit for deploying experiments on a cluster.
## Usage:
e.g.:
```bash
python3 toolkit/run_exp.py --gin ginfile.gin --user username --branch branch_name --gpu-count 4
```

## Help
```
usage: run_exp.py [-h] --gin GIN --user USER --branch BRANCH
                  [--gpu-count GPU_COUNT]
                  [--gpu-type {1080ti,titanx,titanv,rtx2080ti}]
                  [--script SCRIPT] [--install] [--reinstall]

optional arguments:
  -h, --help            show this help message and exit
  --gin GIN             gin file path
  --user USER           username on the cluster
  --branch BRANCH       branch name
  --gpu-count GPU_COUNT
                        number of gpu
  --gpu-type {1080ti,titanx,titanv,rtx2080ti}
                        type of gpu
  --script SCRIPT       custom script
  --install             Install full global venv along with downloading
                        dataset
  --reinstall           Reinstall full global venv - without readownloading
                        dataset
```

### Additional Flags
- Custom script executed before job: `--script=...`
- Install required workspace (only for first run or reinstall): `--install`

### Extra info
- Don't save checkpoints!
- `batcher.batch_size_per_device` should be always `4`
