#!/bin/bash
# use for: nohup sudo docker run -i --entrypoint /bin/sh < script_diffusion.sh  -v /home/jd9169657/simulations:/app -v /home/jd9169657/rareeventestimation/src:/src -v /home/jd9169657/rareeventestimation/docs:/docs rareeventestimation  > log.out 2>&1
python docs/benchmarking/data/enkf-diffusion-problem.py --dir /app/enkf_sim_diffusion
# python docs/benchmarking/enkf-toy-problems.py --dir /app/enkf_sim
python docs/benchmarking/data/sis-diffusion-problem.py --dir /app/sis_sim_diffusion
# python docs/benchmarking/sis-toy-problems.py --dir /app/sis_sim
