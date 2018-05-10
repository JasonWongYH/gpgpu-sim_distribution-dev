#!/usr/bin/python

import argparse
import re
from subprocess import call
import os


parser = argparse.ArgumentParser(description='Process Input Arguments')
parser.add_argument('-n', '--numInput', action="store", type=int, dest='numInput', help="Number of Benchmark we want to run")
parser.add_argument('-c', '--config', action="store", dest='config_loc', help="The name of config file in ./[PTHREAD_BENCHMARK]/configs/")
parser.add_argument('-b', '--benchmarks', action="store", dest='benchmark_list', help="List of benchmarks we are running, separated by comma, - or _")

arg_res = parser.parse_args()

numInput = arg_res.numInput

config_file = arg_res.config_loc
bench_list = re.split('-|_|,', arg_res.benchmark_list)

print "PATH = " + os.environ['PATH']
print "CUDAHOME = " + os.environ['CUDAHOME']
print "CUDA_INSTALL_PATH = " + os.environ['CUDA_INSTALL_PATH']


if(numInput!=len(bench_list)):
    print "ERROR: number of benchmark not matched"
#Note that if this file is called without output argument in condor submit file, it prints the output to STDIO
else:
    # Wait ... We want to actually be able to specify multiple config files, so that condor do not write config over each other. The line below should be avoided
    call(["cp", "/scratch/cluster/rachata/mode3test/quagmire/source/v3.x/pthread_benchmark/configs/"+config_file, "/scratch/cluster/rachata/mode3test/quagmire/source/v3.x/pthread_benchmark/gpgpusim.config"])
    call(["ifconfig"])
    if(numInput == 2):
        call(["/scratch/cluster/rachata/mode3test/quagmire/source/v3.x/pthread_benchmark/gpgpu_ptx_sim__mergedapps", bench_list[0], bench_list[1]])
    elif(numInput == 3):
        call(["/scratch/cluster/rachata/mode3test/quagmire/source/v3.x/pthread_benchmark/gpgpu_ptx_sim__mergedapps", bench_list[0], bench_list[1], bench_list[2]])




