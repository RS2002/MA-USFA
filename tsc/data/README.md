# SUMO network files

The traffic-signal-control experiment runs on the Manhattan 28×7 SUMO scenario.
The training scripts expect these files here:

```
data/raw_data/manhattan_28x7/
  manhattan_28x7.net.xml     road network (196 intersections)
  manhattan_28x7.rou.xml     vehicle routes / demand
```

These SUMO scenario files are not redistributed in this repository. Place your
copy of the Manhattan 28×7 network ([DaRL-LibSignal/LibSignal: MLJ: Libsignal: an open library for traffic signal control](https://github.com/DaRL-LibSignal/LibSignal/tree/master)) under the path above, or edit `DATA_DIR` at
the top of `train_sf.py` and `train.py` to point at your own scenario. Any
`sumo_rl.parallel_env`-compatible `.net.xml` / `.rou.xml` pair will work; the
successor-feature reward and the composition rules are agnostic to the specific
network as long as the five feature terms (queue, wait, pressure, speed,
fairness) can be computed.

Install SUMO from https://www.eclipse.org/sumo/ and make sure `SUMO_HOME` is set
before running the TSC scripts.
