#!/bin/bash
# use for: nohup sudo docker run -i --entrypoint /bin/sh < script_diffusion.sh  -v /home/jd9169657/simulations:/app -v /home/jd9169657/rareeventestimation/src:/src -v /home/jd9169657/rareeventestimation/docs:/docs -d rareeventestimation  > log.out 2>&1
python docs/benchmarking/data/indicator_approximation_performance.py --dir /app/cbree_sim/indicator_functions_performance
# python docs/benchmarking/enkf-toy-problems.py --dir /app/enkf_sim
# python docs/benchmarking/sis-toy-problems.py --dir /app/sis_sim