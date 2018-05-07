ifeq ($(GPGPUSIM_ROOT),)
else
GPGPUSIM_VERSION=$(shell cat $(GPGPUSIM_ROOT)/version | awk '/Version/ {print $$8}' )
GPGPUSIM_BUILD=$(shell cat $(GPGPUSIM_ROOT)/version | awk '/Change/ {print $$6}' )
endif

# Detect CUDA Runtime Version 
CUDA_VERSION_STRING:=$(shell $(CUDA_INSTALL_PATH)/bin/nvcc --version | awk '/release/ {print $$5;}' | sed 's/,//')
CUDART_VERSION:=$(shell echo $(CUDA_VERSION_STRING) | sed 's/\./ /' | awk '{printf("%02u%02u", 10*int($$1), 10*$$2);}')

# Detect GCC Version 
CC_VERSION := $(shell gcc --version | head -1 | awk '{for(i=1;i<=NF;i++){ if(match($$i,/^[0-9]\.[0-9]\.[0-9]$$/))  {print $$i; exit 0 }}}')

# Detect Support for C++11 (C++0x) from GCC Version 
GNUC_CPP0X := $(shell gcc --version | perl -ne 'if (/gcc\s+\(.*\)\s+([0-9.]+)/){ if($$1 >= 4.3) {$$n=1} else {$$n=0;} } END { print $$n; }')

