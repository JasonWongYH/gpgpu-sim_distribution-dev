DEBUG?=1
TRACE?=1

include ../version_detection.mk

CXXFLAGS = -Wall -DDEBUG
CXXFLAGS += -DCUDART_VERSION=$(CUDART_VERSION)

ifeq ($(GNUC_CPP0X), 1)
    CXXFLAGS += -std=c++0x
endif

ifeq ($(TRACE),1)
	CXXFLAGS += -DTRACING_ON=1
endif

ifneq ($(DEBUG),1)
	OPTFLAGS += -O3
else
	CXXFLAGS += 
endif

OPTFLAGS += -g3 -fPIC

CPP = g++ $(SNOW)
OEXT = o

OUTPUT_DIR=$(SIM_OBJ_FILES_DIR)
SRCS = $(shell ls *.cc)
OBJS = $(SRCS:%.cc=$(OUTPUT_DIR)/%.$(OEXT))

$(OUTPUT_DIR)/libgpgpusim.a:	$(OBJS) gpu_uarch_simlib
	ar rcs  $(OUTPUT_DIR)/libgpgpusim.a $(OBJS) $(OUTPUT_DIR)/gpgpu-sim/*.o

gpu_uarch_simlib:
	make   -C ./gpgpu-sim
	
$(OUTPUT_DIR)/Makefile.makedepend: depend

depend:
	touch $(OUTPUT_DIR)/Makefile.makedepend
	makedepend -f$(OUTPUT_DIR)/Makefile.makedepend -p$(OUTPUT_DIR)/ $(SRCS) 2> /dev/null

clean:
	rm -f *.o core *~ *.a Makefile.makedepend Makefile.makedepend.bak

$(OUTPUT_DIR)/%.$(OEXT): %.cc
	$(CPP) $(OPTFLAGS) $(CXXFLAGS) -o $(OUTPUT_DIR)/$*.$(OEXT) -c $*.cc

option_parser.$(OEXT): option_parser.h

include $(OUTPUT_DIR)/Makefile.makedepend

