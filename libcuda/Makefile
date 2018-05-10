include ../version_detection.mk

ifeq ($(OPENGL_SUPPORT),1) 
	GL = -DOPENGL_SUPPORT
endif

ifeq ($(GNUC_CPP0X), 1)
    CXXFLAGS = -std=c++0x
endif


CPP = g++ $(SNOW)
CC = gcc $(SNOW)
CREATELIBRARY    =  1
DEBUG ?= 1
# ifeq ($(DEBUG),1)
CXXFLAGS +=  -Wall -Wno-unused-function -Wno-sign-compare -g -fPIC $(GL)
CCFLAGS += -Wall -Wno-unused-function -Wno-sign-compare -ggdb -fPIC
# else
# 	CXXFLAGS += -O3 -g -Wall -Wno-unused-function -Wno-sign-compare -fPIC $(GL)
# 	CCFLAGS += -Wall -Wno-unused-function -Wno-sign-compare -fPIC
# endif

PROG     =cuda

CXX_SRCS =  cuda_runtime_api.cc
LEX_SRCS = cuobjdump.l
YACC_SRCS = cuobjdump.y
CXXFLAGS += -DCUDART_VERSION=$(CUDART_VERSION)
LEX			= flex
LEXFLAGS	= -B 
YACC		= bison
YFLAGS		= -t -d -v --report=all

.PHONY: clean

OUTPUT_DIR=$(SIM_OBJ_FILES_DIR)/libcuda

OBJS = $(CXX_SRCS:%.cc=$(OUTPUT_DIR)/%.o)
OBJS += $(LEX_SRCS:%.l=$(OUTPUT_DIR)/%_lexer.o)
OBJS += $(YACC_SRCS:%.y=$(OUTPUT_DIR)/%_parser.o)

#--- Make rules ---
lib$(PROG).a: $(OBJS)
	echo $(OBJS)
	ar rcs $(OUTPUT_DIR)/lib$(PROG).a $(OBJS)

$(OUTPUT_DIR)/%.o: %.cc
	$(CPP) $(CXXFLAGS) -I./ -I$(OUTPUT_DIR) -I$(CUDA_INSTALL_PATH)/include  -c $< -o $@

$(OUTPUT_DIR)/%.o: %.c
	$(CC) $(CCFLAGS) -I./ -I$(OUTPUT_DIR) -I$(CUDA_INSTALL_PATH)/include  -c $< -o $@

$(OUTPUT_DIR)/%.o: $(OUTPUT_DIR)/%.c
	$(CC) $(CCFLAGS) -I./ -I$(OUTPUT_DIR) -I$(CUDA_INSTALL_PATH)/include  -c $< -o $@

$(OUTPUT_DIR)/cuobjdump_parser.c: cuobjdump.y
	$(YACC) $(YFLAGS) -p cuobjdump_  -o$@ $< --file-prefix=$(OUTPUT_DIR)/cuobjdump

$(OUTPUT_DIR)/cuobjdump_lexer.c: cuobjdump.l $(OUTPUT_DIR)/cuobjdump_parser.h
	$(LEX) $(LEXFLAGS) -P cuobjdump_ -o$@ $<
	
# The next rule means just get parser.c and you will get parser.h with it
# in other words, get parser.c and do nothing to get parser.h
$(OUTPUT_DIR)/%_parser.h: $(OUTPUT_DIR)/%_parser.c
	:
$(OUTPUT_DIR)/Makefile.makedepend: depend

depend:
	touch $(OUTPUT_DIR)/Makefile.makedepend
	makedepend -f$(OUTPUT_DIR)/Makefile.makedepend -p$(OUTPUT_DIR)/ $(CXX_SRCS) 2> /dev/null

clean:
	rm -f $(PROG)
	rm -f *.o
	rm -f lib$(PROG).a
	rm -f *_parser.*
	rm -f *_lexer.*
	rm -f Makefile.makedepend Makefile.makedepend.bak

include $(OUTPUT_DIR)/Makefile.makedepend

