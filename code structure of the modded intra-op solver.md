# Code Structure of the Modded Intra-op Solver

The specific code of the intra-op solver (a.k.a auto-sharding) is scattered in various files of the project. This page contains some pointers to key components of the intra-op solver and help you navigate the complicated code base. All the modded code is under branch `control-flow` of repo `alpa` and branch `control-flow-support` of repo `tensorflow-alpa`.

## Key Pointers

- Main entrance:
  
  - python entrance(`run_auto_sharding_pass`): https://github.com/HeydrichBeillschmidt/alpa/blob/2d56fa587271e6dc17a54bd0fdd14cc4f6364934/alpa/shard_parallel/auto_sharding.py#L172
  
  - c++ entrance: https://github.com/HeydrichBeillschmidt/tensorflow-alpa/blob/d00f182a00b37e5bd4a77d2bffb8cded32b0d521/tensorflow/compiler/xla/service/spmd/auto_sharding.cc#L2751

- Where the control-flow sharding strategies are registered:
  
  - for while: https://github.com/HeydrichBeillschmidt/tensorflow-alpa/blob/d00f182a00b37e5bd4a77d2bffb8cded32b0d521/tensorflow/compiler/xla/service/spmd/auto_sharding.cc#L1829
  
  - for conditional: https://github.com/HeydrichBeillschmidt/tensorflow-alpa/blob/d00f182a00b37e5bd4a77d2bffb8cded32b0d521/tensorflow/compiler/xla/service/spmd/auto_sharding.cc#L1916

- Where the control-flow sharding strategy settings are enabled:
  
  - config: https://github.com/HeydrichBeillschmidt/alpa/blob/2d56fa587271e6dc17a54bd0fdd14cc4f6364934/alpa/global_env.py#L68
  
  - usage: https://github.com/HeydrichBeillschmidt/alpa/blob/2d56fa587271e6dc17a54bd0fdd14cc4f6364934/alpa/util.py#L852

- Where the ILP solver is called:
  
  - c++ side: https://github.com/HeydrichBeillschmidt/tensorflow-alpa/blob/d00f182a00b37e5bd4a77d2bffb8cded32b0d521/tensorflow/compiler/xla/service/spmd/auto_sharding.cc#L2894
  
  - python side: https://github.com/HeydrichBeillschmidt/alpa/blob/2d56fa587271e6dc17a54bd0fdd14cc4f6364934/alpa/shard_parallel/auto_sharding.py#L590


