help:  # list all targets
	@grep ^[a-z] Makefile

# ros command
init:
	nohup bash init.sh >> /app/logs/init.log 2>&1 &

run-mappo: # e.g. make run-mappo start=1,1 goal=0,0
	nohup bash run_mappo.sh ${start} ${goal} >> /app/logs/mappo_node.log 2>&1 &

clean-node: 
	nohup bash clean_node.sh >> /app/logs/clean_node.log 2>&1 &

rotate: # TODO
	@echo "rotate maxbot to the inital state"
