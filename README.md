# cluster-toolkit
Toolkit for deploying experiments on a cluster.
## usage:
e.g.:
```bash
python3 toolkit/run_exp.py --gin ginfile.gin --user username --branch branch_name --gpu 4
```

### Additional
Custom script executed before job: `--script=...`
Install required workspace (only for first run or reinstall): `--install`
