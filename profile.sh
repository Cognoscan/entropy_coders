#!/bin/bash

perf stat -e resource_stalls.any,resource_stalls.sb,instructions,cpu-clock,cycles,branches,branch-misses,partial_rat_stalls.scoreboard,uops_retired.stall_cycles,uops_executed.stall_cycles,uops_issued.stall_cycles -M tma_core_bound \
	./target/release/deps/fse_benchmark-c4c07cd192e3c49a --bench --profile-time=5

perf stat -e resource_stalls.any,resource_stalls.sb,instructions,cpu-clock,cycles,branches,branch-misses,partial_rat_stalls.scoreboard,uops_retired.stall_cycles,uops_executed.stall_cycles,uops_issued.stall_cycles -M tma_core_bound \
	/home/scott/projects/finitestateentropy-sys/FiniteStateEntropy/programs/fullbench -b9 -P20
