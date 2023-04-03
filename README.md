# Enhancing Multi-Agent Coordination via Dual-Channel Consensus

This is the implementation of the paper "Enhancing Multi-Agent Coordination via Dual-Channel Consensus" written in PyTorch.

## Installation instructions

Set up StarCraft II and SMAC:

```shell
bash install_sc2.sh
```

This will download StarCraft II into the 3rdparty folder and copy the maps necessary to run over. You may also need to set the environment variable for SC2:

```bash
export SC2PATH=[Your SC2 folder like /abc/xyz/3rdparty/StarCraftII]
```

Install Python environment with `requirements.txt` of pip:

```bash
pip install -r requirements.txt

```

## Run an experiment 

```shell
python3 main.py --config=[Algorithm name] --env-config=[Env name] with env_args.map_name=[Map name if choosing SC2 env]
```

The config files act as defaults for an algorithm or environment. 

All results will be stored in the `logs` folder.

For example, run DuCC with QMIX mixing network (default SC2 evaluation in the paper) :

```
python3 main.py --config=qmix --env-config=sc2 with env_args.map_name=3m 
```
