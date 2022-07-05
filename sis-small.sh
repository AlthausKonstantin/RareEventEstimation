#!/bin/bash
# use for: nohup sudo docker run -i --entrypoint /bin/sh < script_diffusion.sh  -v /home/jd9169657/simulations:/app -v /home/jd9169657/rareeventestimation/src:/src -v /home/jd9169657/rareeventestimation/docs:/docs -d rareeventestimation  > log.out 2>&1
python docs/benchmarking/data/sis-toy-problems-small.py --dir /app/sis_sim --counter 21
# python docs/benchmarking/enkf-toy-problems.py --dir /app/enkf_sim
# python docs/benchmarking/sis-toy-problems.py --dir /app/sis_sim