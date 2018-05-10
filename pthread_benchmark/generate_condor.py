import re
import os
from subprocess import call

workload_file = 'twoapp_random.wkld'
config_file_name = "750_noPromotion"

createlog = 1

current_dir = '/scratch/cluster/rachata/mode3test/quagmire/source/v3.x/pthread_benchmark'

#Setup the config
argument_base = "arguments = -c " + config_file_name +" "
output_dir = current_dir +"/results/" + workload_file + "/" + config_file_name +"/"

if(createlog):
    condor_header = "executable     = condor.py\nuniverse       = vanilla \nrequirements = InMastodon\ngetenv = True\n+Group = \"GUEST\"\n+Project = \"ARCHITECTURE\"\n+ProjectDescription = \"MOSAIC Simulation\"\nerror = "+ output_dir + "condor_err/$(Process).err\nlog = "+ output_dir + "condor_log/$(Process).log\n"
else:
    condor_header = "executable     = condor.py\nuniverse       = vanilla \nrequirements = InMastodon\ngetenv = True\n+Group = \"GUEST\"\n+Project = \"ARCHITECTURE\"\n+ProjectDescription = \"MOSAIC Simulation\"\n"

print condor_header




if(not os.path.isdir(output_dir)):
    os.makedirs(output_dir)

if(createlog and not os.path.isdir(output_dir + "/condor_err")):
    os.makedirs(output_dir+ "/condor_err")
if(createlog and not os.path.isdir(output_dir + "/condor_log")):
    os.makedirs(output_dir+ "/condor_log")

call(["cp",current_dir + "/configs/" + config_file_name, output_dir])

output_base = "output = " + output_dir

#Generate the submit script
f = open(current_dir + "/workloads/" + workload_file,'r')
count = 0
for lines in f:
    if count == 0:
        argument_base = argument_base + "-n " + str(len(re.split("_|,|-",lines.split('\n')[0]))) + " "
    else:
        argument = argument_base + "-b " + lines.split('\n')[0] + " "
        print argument
        print output_base + lines.split('\n')[0]
        print "queue"
    count = count + 1
f.close()
