// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ivan Sham,
// Andrew Turner, Ali Bakhoda, The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "gpgpusim_entrypoint.h"
#include <stdio.h>

#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"
#include "cuda-sim/ptx_parser.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/icnt_wrapper.h"
#include "gpgpu-sim/memory_owner.h"
#include "option_parser.h"
#include "stream_manager.h"

#include <pthread.h>
#include <semaphore.h>

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#define DEBUG_ENABLE 0
#define DEBUG_GPU if (DEBUG_ENABLE) \
  std::cout << "GPGPU-Sim: "

struct gpgpu_ptx_sim_arg *grid_params;

sem_t g_sim_signal_start;
sem_t g_sim_signal_finish;
sem_t g_sim_signal_exit;
time_t g_simulation_starttime;
pthread_t g_simulation_thread;

gpgpu_sim_config g_the_gpu_config;
gpgpu_sim *g_the_gpu;
stream_manager *g_stream_manager;

mmu *g_mmu;

static int sg_argc = 3;
static const char *sg_argv[] = {"", "-config", "gpgpusim.config"};

static void print_simulation_time();

pthread_mutex_t g_sim_lock = PTHREAD_MUTEX_INITIALIZER;
bool g_sim_active = false;
bool g_sim_done = true;

void* gpgpu_sim_thread_concurrent(void *) {
  // concurrent kernel execution simulation thread
  do {
    DEBUG_GPU << "*** simulation thread starting and spinning waiting for work ***" << std::endl;
    while (g_stream_manager->empty() && !g_sim_done) {
      ;
    }
    DEBUG_GPU << "GPGPU-Sim: ** START simulation thread (detected work) **" << std::endl <<
        g_stream_manager << std::endl;

    pthread_mutex_lock(&g_sim_lock);
    g_sim_active = true;
    pthread_mutex_unlock(&g_sim_lock);
    bool active = false;
    bool sim_cycles = false;
    DEBUG_GPU << "Initializing GPGPU-sim performance simulator" << std::endl;
    g_the_gpu->init();
    do {
      // check if a kernel has completed
      // launch operation on device if one is pending and can be run

      // Need to break this loop when a kernel completes. This was a
      // source of non-deterministic behaviour in GPGPU-Sim (bug 147).
      // If another stream operation is available, g_the_gpu remains active,
      // causing this loop to not break. If the next operation happens to be
      // another kernel, the gpu is not re-initialized and the inter-kernel
      // behaviour may be incorrect. Check that a kernel has finished and
      // no other kernel is currently running.
      if (g_stream_manager->operation(&sim_cycles) && !g_the_gpu->active())
        break;
      if (g_the_gpu->active()) {
        g_the_gpu->cycle();
        sim_cycles = true;
        g_the_gpu->deadlock_check();
      }
      active = g_the_gpu->active() || !g_stream_manager->empty();
    } while (active);
    DEBUG_GPU << "** STOP simulation thread (no work) **" << std::endl;
    if (sim_cycles) {
      g_the_gpu->update_stats();
      print_simulation_time();
    }
    pthread_mutex_lock(&g_sim_lock);
    g_sim_active = false;
    pthread_mutex_unlock(&g_sim_lock);
  } while (!g_sim_done);
  DEBUG_GPU << "*** simulation thread exiting ***" << std::endl;
  sem_post(&g_sim_signal_exit);
  return NULL;
}

void synchronize() {
  DEBUG_GPU << "GPGPU-Sim: synchronize waiting for inactive GPU simulation" << std::endl <<
      g_stream_manager << std::endl;
  bool done = false;
  do {
    pthread_mutex_lock(&g_sim_lock);
    done = g_stream_manager->empty() && !g_sim_active;
    pthread_mutex_unlock(&g_sim_lock);
  } while (!done);
  DEBUG_GPU << "GPGPU-Sim: detected inactive GPU simulation thread" << std::endl;
}

void exit_simulation() {
  g_sim_done = true;
  DEBUG_GPU << "GPGPU-Sim: exit_simulation called" << std::endl;
  sem_wait(&g_sim_signal_exit);
  DEBUG_GPU << "GPGPU-Sim: simulation thread signaled exit" << std::endl;
}

extern bool g_cuda_launch_blocking;

gpgpu_sim *gpgpu_ptx_sim_init_perf() {
  srand(1);
  print_splash();
  read_sim_environment_variables();
  read_parser_environment_variables();
  option_parser_t opp = option_parser_create();

  icnt_reg_options(opp);
  g_the_gpu_config.reg_options(opp);  // register GPU microrachitecture options
  ptx_reg_options(opp);
  ptx_opcocde_latency_options(opp);
  option_parser_cmdline(opp, sg_argc, sg_argv);  // parse configuration options
  std::cout << "GPGPU-Sim: Configuration options:" << std::endl << std::endl;
  option_parser_print(opp, std::cout);
  std::cout << "GPGPU-Sim: End of Configuration options:" << std::endl
            << std::endl;
  // Set the Numeric locale to a standard locale where a decimal point is a
  // "dot" not a "comma"
  // so it does the parsing correctly independent of the system environment
  // variables
  assert(setlocale(LC_NUMERIC, "C"));

  g_the_gpu_config.init();

  // Rachata: MMU, interface to both cuda-sim and gpgpu-sim
  // FIXME: Can this cause segfault? Something definitely walk into the memory
  // region of g_mmu
  if (g_mmu == NULL) {
    g_mmu = new mmu();
    // TODO: Set the mem_config for mmu object
    g_mmu->init(g_the_gpu_config.get_mem_config());
  }

  g_the_gpu_config.set_mmu(g_mmu);

  g_the_gpu = new gpgpu_sim(g_the_gpu_config);
  g_stream_manager = new stream_manager(g_the_gpu, g_cuda_launch_blocking);
  g_simulation_starttime = time((time_t *)NULL);
  sem_init(&g_sim_signal_start, 0, 0);
  sem_init(&g_sim_signal_finish, 0, 0);
  sem_init(&g_sim_signal_exit, 0, 0);
  return g_the_gpu;
}

void start_sim_thread() {
  if (g_sim_done) {
    g_sim_done = false;
    pthread_create(&g_simulation_thread, NULL, gpgpu_sim_thread_concurrent,
                   NULL);
  }
}

void print_simulation_time() {
  time_t current_time, difference, d, h, m, s;
  current_time = time((time_t *)NULL);
  difference = MAX(current_time - g_simulation_starttime, 1);

  d = difference / (3600 * 24);
  h = difference / 3600 - 24 * d;
  m = difference / 60 - 60 * (h + 24 * d);
  s = difference - 60 * (m + 60 * (h + 24 * d));

  std::cout << std::endl
            << std::endl
            << "gpgpu_simulation_time = " << (unsigned)d << " days, "
            << (unsigned)h << " hrs, " << (unsigned)m << " min, " << (unsigned)s
            << " sec (" << (unsigned)difference << " sec)" << std::endl;
  std::cout << "gpgpu_simulation_rate = "
            << (unsigned)(g_the_gpu->gpu_tot_sim_insn / difference)
            << " (inst/sec)" << std::endl;
  std::cout << "gpgpu_simulation_rate = "
            << (unsigned)(gpu_tot_sim_cycle / difference) << " (cycle/sec)"
            << std::endl;
}

int gpgpu_opencl_ptx_sim_main_perf(kernel_info_t *grid) {
  g_the_gpu->launch(grid);
  sem_post(&g_sim_signal_start);
  sem_wait(&g_sim_signal_finish);
  return 0;
}

//! Functional simulation of OpenCL
/*!
 * This function call the CUDA PTX functional simulator
 */
int gpgpu_opencl_ptx_sim_main_func(kernel_info_t *grid) {
  // calling the CUDA PTX simulator, sending the kernel by reference and a flag
  // set to true,
  // the flag used by the function to distinguish OpenCL calls from the CUDA
  // simulation calls which
  // it is needed by the called function to not register the exit the exit of
  // OpenCL kernel as it doesn't register entering in the first place as the
  // CUDA kernels does
  gpgpu_cuda_ptx_sim_main_func(*grid, true);
  return 0;
}

