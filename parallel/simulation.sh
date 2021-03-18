#!/bin/bash
HOME="/home/ascordeiro"
SIM_HOME=$HOME"/OrCS"
PIN_HOME=$SIM_HOME"/trace_generator/pin"
CODE_HOME=$HOME"/omp_results"
TRACE_HOME=$CODE_HOME"/traces"
THREADS_N=(2 4 8 16 32)

for THREADS in "${THREADS_N[@]}";
do
    case "$1" in
        mlp|MLP)
        	cd $TRACE_HOME
            CONFIG_FILE="configuration_files/sandy_bridge_${THREADS}cores.cfg"
            for inst in 4096 #8192 16384 32768 65536 131072 262144
            do
                for feat in 4096 8192 #8 16 32 64 128 256 512 1024 2048 
                do
                    for i in mlp_avx_${inst}i_${feat}f_${THREADS}t.tid0.stat.out.gz
                    do 
                        cd $SIM_HOME
                        TRACE=mlp_avx_${inst}i_${feat}f_${THREADS}t
                        COUNTER=0
                        COMMAND="./orcs"
                    
                        while [ $COUNTER -lt $THREADS ]; do
                            COMMAND=${COMMAND}' -t '${TRACE_HOME}/${TRACE}
                            let COUNTER=COUNTER+1
                        done
                        nohup ${COMMAND} -c ${CONFIG_FILE} &> ${CODE_HOME}/results/${TRACE}.txt &
                    done
                done 
            done
        ;;
        knn|KNN|kNN)
    		cd $TRACE_HOME
            CONFIG_FILE="configuration_files/sandy_bridge_${THREADS}cores.cfg"
            for inst in 65536 #4096 8192 16384 32768
            do
                for feat in 4096 8192 #8 16 32 64 128 256 512 1024 2048
                do
                    for k in 9 #5 13
                    do
                        for i in knn_avx_${inst}i_${feat}f_${k}k_${THREADS}t.tid0.stat.out.gz
                        do 
                            cd $SIM_HOME
                            TRACE=knn_avx_${inst}i_${feat}f_${k}k_${THREADS}t
                            COUNTER=0
                            COMMAND="./orcs"
                        
                            while [ $COUNTER -lt $THREADS ]; do
                                COMMAND=${COMMAND}' -t '${TRACE_HOME}/${TRACE}
                                let COUNTER=COUNTER+1
                            done
                            nohup ${COMMAND} -c ${CONFIG_FILE} &> ${CODE_HOME}/results/${TRACE}.txt &
                        done
                    done
                done
            done
        ;;
        *)
            echo "Algoritmo inválido!"
        ;;
    esac
done

