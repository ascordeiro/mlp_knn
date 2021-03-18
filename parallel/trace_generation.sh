#!/bin/bash
HOME="/home/ascordeiro"
SIM_HOME=$HOME"/OrCS"
PIN_HOME=$SIM_HOME"/trace_generator/pin"
TRACER_HOME=$SIM_HOME"/trace_generator/extras/pinplay/bin/intel64/sinuca_tracer.so"
CODE_HOME=$HOME"/omp_results"
COMP_FLAGS="-O2 -mavx2 -march=native -fopenmp"
THREADS_N=(2 4 8 16 32)

for THREADS in "${THREADS_N[@]}"
do
    case "$1" in
        mlp|MLP)
            	g++ mlp_avx.c $COMP_FLAGS -o $CODE_HOME/exec/mlp_avx
    		export OMP_NUM_THREADS=${THREADS}
            export OMP_WAIT_POLICY=passive
            for inst in 4096 #8192 16384 32768 65536 131072 262144
            do
                for feat in 4096 8192 #8 16 32 64 128 256 512 1024 2048
                do
                  nohup $PIN_HOME -t $TRACER_HOME -trace x86 -output $CODE_HOME/traces/mlp_avx_${inst}i_${feat}f_${THREADS}t -threads ${THREADS} -- $CODE_HOME/exec/mlp_avx ${inst} ${feat} 2 > nohup.out
                done
            done
        ;;
        knn|KNN|kNN)
    		g++ knn_avx.c $COMP_FLAGS -o $CODE_HOME/exec/knn_avx
    		export OMP_NUM_THREADS=${THREADS}
            export OMP_WAIT_POLICY=passive
            for inst in 65536 #4096 8192 16384 32768
            do
                for feat in 4096 8192 #8 16 32 64 128 256 512 1024 2048
                do
                    for k in 9 #5 9 13
                    do
                       nohup $PIN_HOME -t $TRACER_HOME -trace x86 -output $CODE_HOME/traces/knn_avx_${inst}i_${feat}f_${k}k_${THREADS}t -threads ${THREADS} -- $CODE_HOME/exec/knn_avx ${inst} 256 ${feat} ${k} > nohup.out
                    done
                done
            done
        ;;
        *)
            echo "Algoritmo inválido!"
        ;;
    esac
done 


