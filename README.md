# cluster-toolkit
Toolkit for deploying experiments on a cluster.
## usage:
e.g.:
```bash
python3 toolkit/run_exp.py --gin ginfile.gin --user username --branch branch_name --gpu 4
```

### Additional Flags
- Custom script executed before job: `--script=...`
- Install required workspace (only for first run or reinstall): `--install`

### Extra info
- Don't save checkpoints!
- `batcher.batch_size_per_device` should be always `4`
