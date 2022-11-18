# Build image
     docker build --pull --rm -f "Dockerfile" -t rareeventestimation:latest "." 

# Run a script in container
    docker run  -it --rm -v "$(pwd)"/data/cbree_sim/nonlinear_oscillator:/docs/benchmarking/data/cbree_sim/nonlinear_oscillator rareeventestimation:latest cbree_run_oscillator.sh 

    docker run  -it --rm -v "$(pwd)"/data/cbree_sim/cbree_sim_flowrate:/docs/benchmarking/data/cbree_sim/cbree_sim_flowrate rareeventestimation:latest cbree_run_flowrate.sh

    docker run  -it --rm -v "$(pwd)"/data/enkf_flow_rate:/docs/benchmarking/data/enkf_flow_rate rareeventestimation:latest enkf_run_flowrate.sh
    
    docker run  -it --rm -v "$(pwd)"/data/enkf_sim_oscillator:/docs/benchmarking/data/enkf_sim_oscillator rareeventestimation:latest enkf_run_oscillator.sh 
    
    docker run  -it --rm -v "$(pwd)"/data/sis_flow_rate:/docs/benchmarking/data/sis_flow_rate rareeventestimation:latest sis_run_flowrate.sh
    
    docker run  -it --rm -v "$(pwd)"/data/sis_sim_oscillator:/docs/benchmarking/data/sis_sim_oscillator rareeventestimation:latest sis_run_oscillator.sh


# Run script in container in background

    nnohup sudo docker run --rm -v "$(pwd)"/data/cbree_sim/nonlinear_oscillator:/docs/benchmarking/data/cbree_sim/nonlinear_oscillator rareeventestimation:latest cbree_run_oscillator.sh  &> cbree_oscillator_log 
    
    &ohup sudo docker run --rm -v "$(pwd)"/data/cmc_flow_rate:/docs/benchmarking/data/cmc-flow-rate rareeventestimation:latest cmc_run_oscillator.sh  &> cmc_oscillator_log &