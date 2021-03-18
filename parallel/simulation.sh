]#!/bin/bash
HOME="/home/ascordeiro"
SIM_HOME=$HOME"/OrCS"
PIN_HOME=$SIM_HOME"/trace_generator/pin"
CODE_HOME=$HOME"/omp_results"
TRACE_HOME=$CODE_HOME"/traces"
THREADS_N=(2 4 8 16 32)

for THREADS in "${THREADS_N[@]}";
do
	cd $TRACE_HOME
	CONFIG_FILE="configuration_files/sandy_bridge_${THREADS}cores.cfg"
	for i in *.${THREADS}t.tid0.stat.out.gz
	do 
    	cd $SIM_HOME
    	TRACE=${i%.tid0.stat.out.gz}
    	COUNTER=0
    	COMMAND="./orcs"
    
		while [ $COUNTER -lt $THREADS ]; do
        	COMMAND=${COMMAND}' -t '${TRACE_HOME}/${TRACE}
        	let COUNTER=COUNTER+1
    	done

    	echo "nohup ${COMMAND} -c ${CONFIG_FILE} &> ${CODE_HOME}/resultados/${TRACE}_${DATE_TIME}.txt"
        # nohup ${COMMAND} -c ${CONFIG_FILE} &> ${CODE_HOME}/resultados/${TRACE}_${DATE_TIME}.txt &
	done
done

