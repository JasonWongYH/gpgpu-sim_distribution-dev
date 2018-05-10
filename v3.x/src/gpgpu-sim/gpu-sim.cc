// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, George L. Yuan,
// Ali Bakhoda, Andrew Turner, Ivan Sham
// The University of British Columbia
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

#include "gpu-sim.h"

#include <math.h>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include "zlib.h"

#include "dram.h"
#include "mem_fetch.h"
#include "shader.h"

#include <time.h>
#include "addrdec.h"
#include "delayqueue.h"
#include "dram.h"
#include "gpu-cache.h"
#include "gpu-misc.h"
#include "icnt_wrapper.h"
#include "l2cache.h"
#include "shader.h"
#include "stat-tool.h"

#include "../abstract_hardware_model.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/ptx-stats.h"
#include "../debug.h"
#include "../gpgpusim_entrypoint.h"
#include "../statwrapper.h"
#include "../stream_manager.h"
#include "../trace.h"
#include "mem_latency_stat.h"
#include "power_stat.h"
#include "stats.h"
#include "visualizer.h"

#ifdef GPGPUSIM_POWER_MODEL
#include "power_interface.h"
#else
class gpgpu_sim_wrapper {};
#endif

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <string>

#include "App.h"
#include "Schedules.h"

#define DEBUG_ENABLE 0
#define DEBUG_GPU if (DEBUG_ENABLE) \
  std::cout << "GPGPU-Sim uArch: "
#define DEBUG_SCHED if (1) \
  std::cout << "Debug-Scheduling: "

extern mmu* g_mmu;

unsigned long long gpu_sim_cycle = 0;
int count_tlp = 0;
unsigned long long gpu_tot_sim_cycle = 0;

// performance counter for stalls due to congestion.
unsigned int gpu_stall_dramfull = 0;
unsigned int gpu_stall_icnt2sh = 0;

int my_active_sms = 0;

/* Clock Domains */

#define CORE 0x01
#define L2 0x02
#define DRAM 0x04
#define ICNT 0x08
#define MEM_LATENCY_STAT_IMPL

#include "mem_latency_stat.h"

void power_config::reg_options(class OptionParser* opp) {
  option_parser_register(opp, "-gpuwattch_xml_file", OPT_CSTR,
                         &g_power_config_name, "GPUWattch XML file",
                         "gpuwattch.xml");
  option_parser_register(opp, "-power_simulation_enabled", OPT_BOOL,
                         &g_power_simulation_enabled,
                         "Turn on power simulator (1=On, 0=Off)", "0");
  option_parser_register(opp, "-power_per_cycle_dump", OPT_BOOL,
                         &g_power_per_cycle_dump,
                         "Dump detailed power output each cycle", "0");
  // Output Data Formats
  option_parser_register(
      opp, "-power_trace_enabled", OPT_BOOL, &g_power_trace_enabled,
      "produce a file for the power trace (1=On, 0=Off)", "0");
  option_parser_register(
      opp, "-power_trace_zlevel", OPT_INT32, &g_power_trace_zlevel,
      "Compression level of the power trace output log (0=no comp, 9=highest)",
      "6");
  option_parser_register(
      opp, "-steady_power_levels_enabled", OPT_BOOL,
      &g_steady_power_levels_enabled,
      "produce a file for the steady power levels (1=On, 0=Off)", "0");
  option_parser_register(opp, "-steady_state_definition", OPT_CSTR,
                         &gpu_steady_state_definition,
                         "allowed deviation:number of samples", "8:4");
}

void memory_config::reg_options(class OptionParser* opp) {
  option_parser_register(opp, "-gpgpu_dram_scheduler", OPT_INT32,
                         &scheduler_type, "0 = fifo, 1 = FR-FCFS (defaul)",
                         "1");
  option_parser_register(opp, "-gpgpu_dram_partition_queues", OPT_CSTR,
                         &gpgpu_L2_queue_config, "i2$:$2d:d2$:$2i", "8:8:8:8");
  option_parser_register(opp, "-l2_ideal", OPT_BOOL, &l2_ideal,
                         "Use a ideal L2 cache that always hit", "0");
  option_parser_register(opp, "-gpgpu_cache:dl2", OPT_CSTR,
                         &m_L2_config.m_config_string,
                         "unified banked L2 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>}",
                         "64:128:8,L:B:m:N,A:16:4,4");
  option_parser_register(opp, "-gpgpu_cache:dl2_texture_only", OPT_BOOL,
                         &m_L2_texure_only, "L2 cache used for texture only",
                         "1");
  option_parser_register(
      opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem,
      "number of memory modules (e.g. memory controllers) in gpu", "8");
  option_parser_register(opp, "-gpgpu_n_sub_partition_per_mchannel", OPT_UINT32,
                         &m_n_sub_partition_per_memory_channel,
                         "number of memory subpartition in each memory module",
                         "1");
  option_parser_register(opp, "-gpgpu_n_mem_per_ctrlr", OPT_UINT32,
                         &gpu_n_mem_per_ctrlr,
                         "number of memory chips per memory controller", "1");
  option_parser_register(opp, "-gpgpu_memlatency_stat", OPT_INT32,
                         &gpgpu_memlatency_stat,
                         "track and display latency statistics 0x2 enables MC, "
                         "0x4 enables queue logs",
                         "0");
  option_parser_register(opp, "-gpgpu_frfcfs_dram_sched_queue_size", OPT_INT32,
                         &gpgpu_frfcfs_dram_sched_queue_size,
                         "0 = unlimited (default); # entries per chip", "0");
  option_parser_register(opp, "-gpgpu_dram_return_queue_size", OPT_INT32,
                         &gpgpu_dram_return_queue_size,
                         "0 = unlimited (default); # entries per chip", "0");
  option_parser_register(opp, "-gpgpu_num_groups", OPT_INT32, &gpgpu_num_groups,
                         "number of containers (application equal partitions)",
                         "2");
  option_parser_register(opp, "-gpgpu_dram_buswidth", OPT_UINT32, &busW,
                         "default = 4 bytes (8 bytes per cycle at DDR)", "4");
  option_parser_register(
      opp, "-gpgpu_dram_burst_length", OPT_UINT32, &BL,
      "Burst length of each DRAM request (default = 4 data bus cycle)", "4");
  option_parser_register(opp, "-dram_data_command_freq_ratio", OPT_UINT32,
                         &data_command_freq_ratio,
                         "Frequency ratio between DRAM data bus and command "
                         "bus (default = 2 times, i.e. DDR)",
                         "2");
  option_parser_register(
      opp, "-gpgpu_dram_timing_opt", OPT_CSTR, &gpgpu_dram_timing_opt,
      "DRAM timing parameters = "
      "{nbk:tCCD:tRRD:tRCD:tRAS:tRP:tRC:CL:WL:tCDLR:tWR:nbkgrp:tCCDL:tRTPL}",
      "4:2:8:12:21:13:34:9:4:5:13:1:0:0");
  option_parser_register(opp, "-gpgpu_subarray_timing_opt", OPT_CSTR,
                         &gpgpu_dram_subarray_timing_opt,
                         "DRAM subarray timing parameters = "
                         "{nsa:sCCD:sRRD:sRCD:sRAS:sRP:sRC:sCL:sWL:sCDLR:sWR:"
                         "sCCDL:sRTPL}",
                         "16:2:8:12:21:13:34:9:4:5:13:0:0");
  option_parser_register(opp, "-rop_latency", OPT_UINT32, &rop_latency,
                         "ROP queue latency (default 85)", "85");
  option_parser_register(opp, "-dram_latency", OPT_UINT32, &dram_latency,
                         "DRAM latency (default 30)", "30");
  option_parser_register(opp, "-gpu_char", OPT_INT32, &gpu_char,
                         "gpu char activated", "0");
  // Page/TLB studies
  option_parser_register(opp, "-page_queue_size", OPT_UINT32, &page_queue_size,
                         "Size of the page queue", "2048");
  option_parser_register(opp, "-tlb_flush_enable", OPT_UINT32,
                         &TLB_flush_enable, "Enabling TLB flush", "0");
  option_parser_register(
      opp, "-get_shader_warp_stat", OPT_UINT32, &get_shader_avail_warp_stat,
      "Collect number of schedulable warps statistics in shader.cc", "0");
  option_parser_register(
      opp, "-tlb_flush_freq", OPT_UINT32, &TLB_flush_freq,
      "If TLB flush is enabled, what is the frequency (per L1 cache accesse)",
      "10000000");
  option_parser_register(
      opp, "-page_stat_update_cycle", OPT_UINT32, &page_stat_update_cycle,
      "How often superpage stats are cleared (Default = 100k cycles)",
      "100000");
  option_parser_register(opp, "-demotion_check_cycle", OPT_UINT32,
                         &demotion_check_cycle,
                         "How often demotion is being checked for clearing "
                         "(Default = 100k cycles)",
                         "100000");
  option_parser_register(
      opp, "-l1_tlb_invalidate_latency", OPT_UINT32, &l1_tlb_invalidate_latency,
      "Latency for L1 TLB invalidation (Default = 15 cycles)", "15");
  option_parser_register(
      opp, "-l2_tlb_invalidate_latency", OPT_UINT32, &l2_tlb_invalidate_latency,
      "Latency for L1 TLB invalidation (Default = 200 cycles)", "200");
  option_parser_register(opp, "-enable_PCIe", OPT_BOOL, &enable_PCIe,
                         "Enable PCIe latency (otherwise PCIe latency = 0)",
                         "false");
  option_parser_register(opp, "-capture_VA", OPT_BOOL, &capture_VA,
                         "Tracing Virtual Address from the runs (true/false)",
                         "false");
  option_parser_register(opp, "-va_trace_file", OPT_CSTR, &va_trace_file,
                         "Output file of the virtual address", "VA.trace");
  option_parser_register(opp, "-va_mask", OPT_CSTR, &va_mask,
                         "Mask of the virtual address for PT walk routine "
                         "(should match with tlb_levels)",
                         "11111222223333344444000000000000");
  option_parser_register(
      opp, "-page_mapping_policy", OPT_UINT32, &page_mapping_policy,
      "VA to PA mapping policy (default is 1 (random), 0 = for debugging", "1");
  option_parser_register(opp, "-pw_cache_enable", OPT_BOOL, &pw_cache_enable,
                         "Enabling PW cache (0 = false, 1 = true)", "0");
  option_parser_register(opp, "-enable_subarray", OPT_UINT32, &enable_subarray,
                         "Enabling Subarray (0 = false, 1 = true)", "0");
  option_parser_register(
      opp, "-channel_partition", OPT_UINT32, &channel_partition,
      "Enabling Channel Partitioning (0 = false, 1 = policy 1, etc.)", "0");
  // Copy/Zero
  option_parser_register(opp, "-RC_enabled", OPT_UINT32, &RC_enabled,
                         "Enable Row Clone", "0");
  option_parser_register(opp, "-LISA_enabled", OPT_UINT32, &LISA_enabled,
                         "Enable LISA", "0");
  option_parser_register(opp, "-MASA_enabled", OPT_UINT32, &MASA_enabled,
                         "Enabling Subarray (0 = false, 1 = true)", "0");
  option_parser_register(opp, "-SALP_enabled", OPT_UINT32, &SALP_enabled,
                         "Enabling Subarray (0 = false, 1 = true)", "0");
  option_parser_register(opp, "-interSA_latency", OPT_UINT32, &interSA_latency,
                         "Inter subarray copy latency", "50");
  option_parser_register(opp, "-intraSA_latency", OPT_UINT32, &intraSA_latency,
                         "Intra subarray copy latency", "1000");
  option_parser_register(opp, "-lisa_latency", OPT_UINT32, &lisa_latency,
                         "Intra subarray copy latency using LISA", "100");
  option_parser_register(opp, "-RCintraSA_latency", OPT_UINT32,
                         &RCintraSA_latency,
                         "Intra subarray copy latency using RC", "1000");
  option_parser_register(opp, "-RCzero_latency", OPT_UINT32, &RCzero_latency,
                         "Zero a page latency using RC", "100");
  option_parser_register(opp, "-zero_latency", OPT_UINT32, &zero_latency,
                         "Zeroing a page latency", "1000");
  option_parser_register(opp, "-interBank_latency", OPT_UINT32,
                         &interBank_latency, "Copy a page across bank", "1000");
  option_parser_register(opp, "-RCpsm_latency", OPT_UINT32, &RCpsm_latency,
                         "Copy a page across bank using RC psm", "1000");
  option_parser_register(
      opp, "-bank_partition", OPT_UINT32, &bank_partition,
      "Enabling Bank Partitioning (0 = false, 1 = policy 1, etc.)", "0");
  option_parser_register(
      opp, "-subarray_partition", OPT_UINT32, &subarray_partition,
      "Enabling Subarray Partitioning (0 = false, 1 = policy 1, etc.)", "0");
  option_parser_register(opp, "-pw_cache_latency", OPT_UINT32,
                         &pw_cache_latency, "PW cache latency", "10");
  option_parser_register(opp, "-pw_cache_num_ports", OPT_UINT32,
                         &pw_cache_num_ports,
                         "Number of ports for the PW cache", "4");
  option_parser_register(opp, "-tlb_pw_cache_entries", OPT_UINT32,
                         &tlb_pw_cache_entries, "Number of PW cache entries",
                         "64");
  option_parser_register(opp, "-tlb_pw_cache_ways", OPT_UINT32,
                         &tlb_pw_cache_ways, "Number of PW cache ways", "8");
  option_parser_register(
      opp, "-tlb_replacement_policy", OPT_UINT32, &tlb_replacement_policy,
      "TLB Replacement policy (0 = LRU (default), 1 = WID based", "0");
  option_parser_register(
      opp, "-tlb_replacement_hash_size", OPT_UINT32, &tlb_replacement_hash_size,
      "TLB Replacement policy hashed size for policy 3 and 4", "127");
  option_parser_register(opp, "-tlb_replacement_high_threshold", OPT_UINT32,
                         &tlb_replacement_high_threshold,
                         "More than XX WID share this entry", "20");
  option_parser_register(opp, "-tlb_core_index", OPT_UINT32, &tlb_core_index,
                         "Scramble coreID to TLB Index bits", "0");
  option_parser_register(opp, "-tlb_prefetch", OPT_UINT32, &tlb_prefetch,
                         "Enabling TLB Prefetch", "0");
  option_parser_register(
      opp, "-tlb_fixed_latency_enabled", OPT_UINT32, &tlb_fixed_latency_enabled,
      "Enabling TLB Fixed latency instead of consecutive DRAM requests "
      "(0=disable, 1=enable (applies to TLB miss))",
      "0");
  option_parser_register(
      opp, "-tlb_fixed_latency", OPT_UINT32, &tlb_fixed_latency,
      "If TLB latency is a fixed number, what is the latency (default = 100)",
      "100");
  option_parser_register(
      opp, "-tlb_L1_flush_cycles", OPT_UINT32, &tlb_L1_flush_cycles,
      "How long L1 TLB is blocked during TLB flush (default = 1000)", "1000");
  option_parser_register(
      opp, "-tlb_L2_flush_cycles", OPT_UINT32, &tlb_L2_flush_cycles,
      "How long L2 TLB is blocked during TLB flush (default = 50000)", "50000");
  option_parser_register(opp, "-tlb_bypass_cache_flush_cycles", OPT_UINT32,
                         &tlb_bypass_cache_flush_cycles,
                         "How long bypass cache in L2 TLB is blocked during "
                         "TLB flush (default = 1000)",
                         "1000");
  option_parser_register(
      opp, "-tlb_prefetch_set", OPT_UINT32, &tlb_prefetch_set,
      "If prefetch policy = 3, how many prefetch buffer set", "16");
  option_parser_register(opp, "-tlb_prefetch_buffer_size", OPT_UINT32,
                         &tlb_prefetch_buffer_size,
                         "TLB Prefetch buffer size (per core)", "16");
  option_parser_register(opp, "-capture_VA_map", OPT_BOOL, &capture_VA_map,
                         "Tracing Virtual Address from the runs (true/false)",
                         "false");
  option_parser_register(opp, "-pt_file", OPT_CSTR, &pt_file,
                         "Input page table trace", "pt_map.trace");
  option_parser_register(opp, "-epoch_length", OPT_UINT32, &epoch_length,
                         "Stat collection epoch length", "10000");
  option_parser_register(opp, "-epoch_length", OPT_BOOL, &epoch_enabled,
                         "Stat collection epoch (true/false), default - true",
                         "true");
  option_parser_register(opp, "-tlb_cache_depth", OPT_UINT32,
                         &max_tlb_cache_depth,
                         "How many levels is the TLB cache", "0");
  option_parser_register(opp, "-tlb_victim_size", OPT_UINT32, &tlb_victim_size,
                         "Size of the TLB victim cache", "32");
  option_parser_register(opp, "-tlb_victim_size_large", OPT_UINT32,
                         &tlb_victim_size_large,
                         "Size of the TLB victim cache for huge page", "8");
  option_parser_register(opp, "-l2_tlb_ways", OPT_UINT32, &l2_tlb_ways,
                         "L2 TLB ways total (shared)", "16");
  option_parser_register(opp, "-l2_tlb_ways_large", OPT_UINT32,
                         &l2_tlb_ways_large,
                         "L2 huge page TLB ways total (shared)", "8");
  option_parser_register(
      opp, "-l2data_tlb_way_reset", OPT_UINT32, &l2_tlb_way_reset,
      "When will L2 data TLB way yield to normal data", "100000");
  option_parser_register(opp, "-tlb_cache_part", OPT_UINT32, &tlb_cache_part,
                         "L2 partitioning for TLB", "0");
  option_parser_register(opp, "-l2_tlb_entries", OPT_UINT32, &l2_tlb_entries,
                         "L2 TLB entires (total entries) (shared)", "1024");
  option_parser_register(
      opp, "-l2_tlb_entries_large", OPT_UINT32, &l2_tlb_entries_large,
      "L2 TLB entires for huge page (total entries) (shared)", "256");
  option_parser_register(opp, "-tlb_lookup_bypass", OPT_UINT32,
                         &tlb_lookup_bypass, "Bypass TLB lookup", "0");
  option_parser_register(opp, "-tlb_miss_rate_ratio", OPT_UINT32,
                         &tlb_miss_rate_ratio,
                         "Safety need to avoid ping-ponging between not bypass "
                         "and bypass TLB lookup",
                         "2");
  option_parser_register(opp, "-tlb_stat_resets", OPT_UINT32, &tlb_stat_resets,
                         "Reassign tokens every XX", "2000");
  option_parser_register(opp, "-max_tlb_miss", OPT_UINT32, &max_tlb_miss,
                         "Kicks in TLB bypass when TLB miss of (in percentage)",
                         "5");
  option_parser_register(opp, "-tlb_bypass_initial_tokens", OPT_UINT32,
                         &tlb_bypass_initial_tokens,
                         "Number of tokens available in TLB bypassing", "1000");
  option_parser_register(opp, "-tlb_high_prio_level", OPT_UINT32,
                         &tlb_high_prio_level,
                         "At which level tlb gets more priority in DRAM", "2");
  option_parser_register(opp, "-tlb_dram_aware", OPT_UINT32, &tlb_dram_aware,
                         "DRAM treat tlb differently", "0");
  option_parser_register(
      opp, "-dram_switch_factor", OPT_UINT32, &dram_switch_factor,
      "DRAM policy 5, how often apps prioritization is switched (random factor "
      "= this_factor/100 ops (bw_factor), default = 100)",
      "100");
  option_parser_register(opp, "-dram_switch_max", OPT_UINT32, &dram_switch_max,
                         "DRAM policy 5, maximum before DRAM sched switch the "
                         "app, default = 1000)",
                         "1000");
  option_parser_register(opp, "-dram_switch_threshold", OPT_UINT32,
                         &dram_switch_threshold,
                         "DRAM policy 5, threshold before DRAM sched switch "
                         "the app, default = 100)",
                         "100");
  option_parser_register(
      opp, "-dram_high_prio_chance", OPT_UINT32, &dram_high_prio_chance,
      "DRAM policy 6, how likely a data request goes into high prio queue "
      "(probability = concurrent_request/this number, default = 100)",
      "100");
  option_parser_register(
      opp, "-dram_scheduling_policy", OPT_UINT32, &dram_scheduling_policy,
      "DRAM scheduling policy (new for MASK), 0 = FR-FCFS, 1=FCFS", "0");
  option_parser_register(opp, "-max_DRAM_high_prio_wait", OPT_UINT32,
                         &max_DRAM_high_prio_wait,
                         "DRAM row coalescing for high_prio queue", "100");
  option_parser_register(
      opp, "-max_DRAM_high_prio_combo", OPT_UINT32, &max_DRAM_high_prio_combo,
      "How many consecutive high prio requests are issued", "8");
  option_parser_register(opp, "-dram_batch", OPT_BOOL, &dram_batch,
                         "Batch high priority DRAM requests (true/false)",
                         "false");
  option_parser_register(
      opp, "-page_transfer_time", OPT_UINT32, &page_transfer_time,
      "PCIe latency to transfer a page (default = 1000)", "1000");
  option_parser_register(opp, "-tlb_size", OPT_UINT32, &tlb_size,
                         "Size of TLB per SM", "64");
  option_parser_register(opp, "-tlb_size_large", OPT_UINT32, &tlb_size_large,
                         "Size of huge page TLB per SM", "16");
  option_parser_register(opp, "-tlb_prio_max_level", OPT_UINT32,
                         &tlb_prio_max_level,
                         "Max level of TLBs that gets high priority", "0");
  // Parse from va_mask (or page_size_list) instead
  //    option_parser_register(opp, "-tlb_levels", OPT_UINT32, &tlb_levels,
  //                 "Number of VA to PA levels", "4");
  option_parser_register(opp, "-tlb_bypass_enabled", OPT_UINT32,
                         &tlb_bypass_enabled,
                         "Bypass L2 Cache for TLB requests (0 = Disable, 1 = "
                         "Static policy, 2 = dynamic policy using threshold)",
                         "0");
  option_parser_register(opp, "-tlb_bypass_level", OPT_UINT32,
                         &tlb_bypass_level,
                         "Bypass L2 cache for TLB level starting at N", "2");
  option_parser_register(opp, "-data_cache_bypass_threshold", OPT_UINT32,
                         &data_cache_bypass_threshold,
                         "Threshold used for bypassing L2 data cache for "
                         "TLB-related requests (in percentage) for TLB L2 data "
                         "cache bypass policy 2",
                         "80");
  option_parser_register(opp, "-enable_page_coalescing", OPT_UINT32,
                         &enable_page_coalescing,
                         "Enable page coalescing (default = 0)", "0");
  option_parser_register(opp, "-enable_compaction", OPT_UINT32,
                         &enable_compaction, "Enable compaction (default = 0)",
                         "0");
  option_parser_register(
      opp, "-enable_rctest", OPT_UINT32, &enable_rctest,
      "If set to 1, randomly send rowclone commands for testing (default = 0)",
      "0");
  option_parser_register(
      opp, "-compaction_probe_cycle", OPT_UINT32, &compaction_probe_cycle,
      "How frequency compaction probes DRAM (default = 100000)", "100000");
  option_parser_register(
      opp, "-compaction_probe_additional_latency", OPT_UINT32,
      &compaction_probe_additional_latency,
      "Additional latency when probing, for testing purposes (default = 0)",
      "0");
  option_parser_register(
      opp, "-enable_costly_coalesce", OPT_UINT32, &enable_costly_coalesce,
      "Enable page coalescing even when there are pages from other app within "
      "the coalesce range (default = 0)",
      "0");
  option_parser_register(opp, "-page_coalesce_locality_thres", OPT_UINT32,
                         &page_coalesce_locality_thres,
                         "Threshold for locality (number of pages touched in "
                         "the large page range, in percentage) to trigger page "
                         "coalescing (default = 101 (i.e., impossible to "
                         "happen)",
                         "101");
  option_parser_register(opp, "-page_coalesce_hotness_thres", OPT_UINT32,
                         &page_coalesce_hotness_thres,
                         "Threshold for hotness (how long this page is "
                         "touched) before trigger coalescing, default = 100",
                         "100");
  option_parser_register(opp, "-page_coalesce_lower_thres_offset", OPT_UINT32,
                         &page_coalesce_lower_thres_offset,
                         "The gap in percentage between coalesce/demotion "
                         "threshold in percentage, default = 10 percent of the "
                         "threshold",
                         "10");
  option_parser_register(opp, "-tlb_enabled", OPT_UINT32, &tlb_enable,
                         "TLB and PT_walk enable (true/false)", "1");
  // Set automatically using the page_size_list
  //    option_parser_register(opp, "-base_page_size", OPT_UINT32,
  //    &base_page_size,
  //                 "Size of a base page (default = 4096)", "4096");
  option_parser_register(opp, "-page_size_list", OPT_CSTR, &page_size_list,
                         "List of differing page sizes (in bytes) (default = "
                         "2MB and 4KB page sizes)",
                         "2097152:4096");
  //    option_parser_register(opp, "-page_size", OPT_UINT32, &page_size,
  //                 "Size of a page (factor of 2, 12 means 2^12)", "12");
  option_parser_register(
      opp, "-dram_row_size", OPT_UINT32, &dram_row_size,
      "Size of a dram row buffer (factor of 2, 12 means 2^12, default = 4KB)",
      "12");
  option_parser_register(opp, "-DRAM_size", OPT_UINT32, &DRAM_size,
                         "Size of the DRAM", "3221225472");
  option_parser_register(opp, "-DRAM_fragmentation", OPT_FLOAT,
                         &DRAM_fragmentation,
                         "Initial fragmentation as percent of the DRAM", "0.0");
  option_parser_register(opp, "-DRAM_fragmentation_pages_per_frame", OPT_UINT32,
                         &DRAM_fragmentation_pages_per_frame,
                         "Number of pages to add to a fragmented frame", "1");
  //    option_parser_register(opp, "-tlb_template_bits", OPT_UINT32,
  //    &tlb_template_bits,
  //                 "Template bits for all TLB's memory accesses",
  //                 "4294967296");
  //    option_parser_register(opp, "-page_appID_bits", OPT_UINT32,
  //    &page_appID_bits,
  //                 "Location of the appID bits", "28");
  //    option_parser_register(opp, "-page_tlb_level_bits", OPT_UINT32,
  //    &page_tlb_level_bits,
  //                 "Location of the tlb_access_level bit", "5");
  option_parser_register(
      opp, "-page_evict_policy", OPT_UINT32, &page_evict_policy,
      "Page eviction policy (0=app-LRU (default), 1=global-LRU, 2=victim)",
      "0");
  option_parser_register(opp, "-page_partition_policy", OPT_UINT32,
                         &page_partition_policy,
                         "DRAM partitioning policy (default = 0)", "0");
  option_parser_register(opp, "-PCIe_queue_size", OPT_UINT32, &PCIe_queue_size,
                         "Size of the PCIe queue", "2048");
  m_address_mapping.addrdec_setoption(opp);
}

void shader_core_config::reg_options(class OptionParser* opp) {
  option_parser_register(opp, "-gpgpu_simd_model", OPT_INT32, &model,
                         "1 = post-dominator", "1");
  option_parser_register(
      opp, "-gpgpu_shader_core_pipeline", OPT_CSTR,
      &gpgpu_shader_core_pipeline_opt,
      "shader core pipeline config, i.e., {<nthread>:<warpsize>}", "1024:32");
  option_parser_register(opp, "-gpgpu_tex_cache:l1", OPT_CSTR,
                         &m_L1T_config.m_config_string,
                         "per-shader L1 texture cache  (READ-ONLY) config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>:<rf>}",
                         "8:128:5,L:R:m:N,F:128:4,128:2");
  option_parser_register(
      opp, "-gpgpu_const_cache:l1", OPT_CSTR, &m_L1C_config.m_config_string,
      "per-shader L1 constant memory cache  (READ-ONLY) config "
      " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<"
      "merge>,<mq>} ",
      "64:64:2,L:R:f:N,A:2:32,4");
  option_parser_register(opp, "-gpgpu_cache:il1", OPT_CSTR,
                         &m_L1I_config.m_config_string,
                         "shader L1 instruction cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>} ",
                         "4:256:4,L:R:f:N,A:2:32,4");
  option_parser_register(opp, "-gpgpu_cache:dl1", OPT_CSTR,
                         &m_L1D_config.m_config_string,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_cache:dl1PrefL1", OPT_CSTR,
                         &m_L1D_config.m_config_stringPrefL1,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_cache:dl1PreShared", OPT_CSTR,
                         &m_L1D_config.m_config_stringPrefShared,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gmem_skip_L1D", OPT_BOOL, &gmem_skip_L1D,
                         "global memory access skip L1D cache (implements "
                         "-Xptxas -dlcm=cg, default=no skip)",
                         "0");
  option_parser_register(opp, "-gpgpu_perfect_mem", OPT_BOOL,
                         &gpgpu_perfect_mem,
                         "enable perfect memory mode (no cache miss)", "0");
  option_parser_register(
      opp, "-n_regfile_gating_group", OPT_UINT32, &n_regfile_gating_group,
      "group of lanes that should be read/written together)", "4");
  option_parser_register(
      opp, "-gpgpu_clock_gated_reg_file", OPT_BOOL, &gpgpu_clock_gated_reg_file,
      "enable clock gated reg file for power calculations", "0");
  option_parser_register(
      opp, "-gpgpu_clock_gated_lanes", OPT_BOOL, &gpgpu_clock_gated_lanes,
      "enable clock gated lanes for power calculations", "0");
  option_parser_register(opp, "-gpgpu_shader_registers", OPT_UINT32,
                         &gpgpu_shader_registers,
                         "Number of registers per shader core. Limits number "
                         "of concurrent CTAs. (default 8192)",
                         "8192");
  option_parser_register(
      opp, "-gpgpu_shader_cta", OPT_UINT32, &max_cta_per_core,
      "Maximum number of concurrent CTAs in shader (default 8)", "8");
  option_parser_register(opp, "-gpgpu_n_clusters", OPT_UINT32, &n_simt_clusters,
                         "number of processing clusters", "10");
  option_parser_register(opp, "-gpgpu_n_cores_per_cluster", OPT_UINT32,
                         &n_simt_cores_per_cluster,
                         "number of simd cores per cluster", "3");
  option_parser_register(opp, "-gpgpu_n_cluster_ejection_buffer_size",
                         OPT_UINT32, &n_simt_ejection_buffer_size,
                         "number of packets in ejection buffer", "8");
  option_parser_register(
      opp, "-gpgpu_n_ldst_response_buffer_size", OPT_UINT32,
      &ldst_unit_response_queue_size,
      "number of response packets in ld/st unit ejection buffer", "2");
  option_parser_register(
      opp, "-gpgpu_shmem_size", OPT_UINT32, &gpgpu_shmem_size,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(
      opp, "-gpgpu_shmem_size", OPT_UINT32, &gpgpu_shmem_sizeDefault,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(
      opp, "-gpgpu_shmem_size_PrefL1", OPT_UINT32, &gpgpu_shmem_sizePrefL1,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(opp, "-gpgpu_shmem_size_PrefShared", OPT_UINT32,
                         &gpgpu_shmem_sizePrefShared,
                         "Size of shared memory per shader core (default 16kB)",
                         "16384");
  option_parser_register(
      opp, "-gpgpu_shmem_num_banks", OPT_UINT32, &num_shmem_bank,
      "Number of banks in the shared memory in each shader core (default 16)",
      "16");
  option_parser_register(
      opp, "-gpgpu_shmem_limited_broadcast", OPT_BOOL, &shmem_limited_broadcast,
      "Limit shared memory to do one broadcast per cycle (default on)", "1");
  option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32,
                         &mem_warp_parts,
                         "Number of portions a warp is divided into for shared "
                         "memory bank conflict check ",
                         "2");
  option_parser_register(
      opp, "-gpgpu_warpdistro_shader", OPT_INT32, &gpgpu_warpdistro_shader,
      "Specify which shader core to collect the warp size distribution from",
      "-1");
  option_parser_register(
      opp, "-gpgpu_warp_issue_shader", OPT_INT32, &gpgpu_warp_issue_shader,
      "Specify which shader core to collect the warp issue distribution from",
      "0");
  option_parser_register(opp, "-gpgpu_local_mem_map", OPT_BOOL,
                         &gpgpu_local_mem_map,
                         "Mapping from local memory space address to simulated "
                         "GPU physical address space (default = enabled)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_reg_banks", OPT_INT32,
                         &gpgpu_num_reg_banks,
                         "Number of register banks (default = 8)", "8");
  option_parser_register(
      opp, "-gpgpu_reg_bank_use_warp_id", OPT_BOOL, &gpgpu_reg_bank_use_warp_id,
      "Use warp ID in mapping registers to banks (default = off)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_units_sp,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_units_sfu,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_units_mem,
                         "number of collector units (default = 2)", "2");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_units_gen,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_sp,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_sfu,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_mem,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_gen,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_sp,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_sfu,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_mem,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_gen,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_coalesce_arch", OPT_INT32, &gpgpu_coalesce_arch,
      "Coalescing arch (default = 13, anything else is off for now)", "13");
  option_parser_register(opp, "-gpgpu_num_sched_per_core", OPT_INT32,
                         &gpgpu_num_sched_per_core,
                         "Number of warp schedulers per core", "1");
  option_parser_register(opp, "-gpgpu_max_insn_issue_per_warp", OPT_INT32,
                         &gpgpu_max_insn_issue_per_warp,
                         "Max number of instructions that can be issued per "
                         "warp in one cycle by scheduler",
                         "2");
  option_parser_register(opp, "-gpgpu_simt_core_sim_order", OPT_INT32,
                         &simt_core_sim_order,
                         "Select the simulation order of cores in a cluster "
                         "(0=Fix, 1=Round-Robin)",
                         "1");
  option_parser_register(
      opp, "-gpgpu_pipeline_widths", OPT_CSTR, &pipeline_widths_string,
      "Pipeline widths "
      "ID_OC_SP,ID_OC_SFU,ID_OC_MEM,OC_EX_SP,OC_EX_SFU,OC_EX_MEM,EX_WB",
      "1,1,1,1,1,1,1");
  option_parser_register(opp, "-gpgpu_num_sp_units", OPT_INT32,
                         &gpgpu_num_sp_units, "Number of SP units (default=1)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_sfu_units", OPT_INT32,
                         &gpgpu_num_sfu_units, "Number of SF units (default=1)",
                         "1");
  option_parser_register(
      opp, "-gpgpu_num_mem_units", OPT_INT32, &gpgpu_num_mem_units,
      "Number if ldst units (default=1) WARNING: not hooked up to anything",
      "1");
  option_parser_register(
      opp, "-gpgpu_scheduler", OPT_CSTR, &gpgpu_scheduler_string,
      "Scheduler configuration: < lrr | gto | two_level_active > "
      "If "
      "two_level_active:<num_active_warps>:<inner_prioritization>:<outer_"
      "prioritization>"
      "For complete list of prioritization values see shader.h enum "
      "scheduler_prioritization_type"
      "Default: gto",
      "gto");
}

void gpgpu_sim_config::reg_options(option_parser_t opp) {
  gpgpu_functional_sim_config::reg_options(opp);
  m_shader_config.reg_options(opp);
  m_memory_config.reg_options(opp);
  power_config::reg_options(opp);
  option_parser_register(opp, "-gpgpu_max_cycle", OPT_INT32, &gpu_max_cycle_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_insn", OPT_INT32, &gpu_max_insn_opt,
                         "terminates gpu simulation early (0 = no limit)",
                         "1000000000");  // 500M
  option_parser_register(opp, "-gpgpu_max_cta", OPT_INT32, &gpu_max_cta_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(
      opp, "-gpgpu_runtime_stat", OPT_CSTR, &gpgpu_runtime_stat,
      "display runtime statistics such as dram utilization {<freq>:<flag>}",
      "10000:0");
  option_parser_register(opp, "-liveness_message_freq", OPT_INT64,
                         &liveness_message_freq,
                         "Minimum number of seconds between simulation "
                         "liveness messages (0 = always print)",
                         "1");
  option_parser_register(opp, "-gpgpu_flush_l1_cache", OPT_BOOL,
                         &gpgpu_flush_l1_cache,
                         "Flush L1 cache at the end of each kernel call", "0");
  option_parser_register(opp, "-gpgpu_flush_l2_cache", OPT_BOOL,
                         &gpgpu_flush_l2_cache,
                         "Flush L2 cache at the end of each kernel call", "0");
  option_parser_register(
      opp, "-gpgpu_deadlock_detect", OPT_BOOL, &gpu_deadlock_detect,
      "Stop the simulation at deadlock (1=on (default), 0=off)", "1");
  option_parser_register(opp, "-gpgpu_ptx_instruction_classification",
                         OPT_INT32, &gpgpu_ptx_instruction_classification,
                         "if enabled will classify ptx instruction types per "
                         "kernel (Max 255 kernels now)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_ptx_sim_mode", OPT_INT32, &g_ptx_sim_mode,
      "Select between Performance (default) or Functional simulation (1)", "0");
  option_parser_register(opp, "-gpgpu_clock_domains", OPT_CSTR,
                         &gpgpu_clock_domains,
                         "Clock Domain Frequencies in MhZ {<Core Clock>:<ICNT "
                         "Clock>:<L2 Clock>:<DRAM Clock>}",
                         "500.0:2000.0:2000.0:2000.0");
  option_parser_register(
      opp, "-gpgpu_max_concurrent_kernel", OPT_INT32, &max_concurrent_kernel,
      "maximum kernels that can run concurrently on GPU", "8");
  option_parser_register(
      opp, "-gpgpu_cflog_interval", OPT_INT32, &gpgpu_cflog_interval,
      "Interval between each snapshot in control flow logger", "0");
  option_parser_register(opp, "-visualizer_enabled", OPT_BOOL,
                         &g_visualizer_enabled,
                         "Turn on visualizer output (1=On, 0=Off)", "1");
  option_parser_register(opp, "-visualizer_outputfile", OPT_CSTR,
                         &g_visualizer_filename,
                         "Specifies the output log file for visualizer", NULL);
  option_parser_register(
      opp, "-visualizer_zlevel", OPT_INT32, &g_visualizer_zlevel,
      "Compression level of the visualizer output log (0=no comp, 9=highest)",
      "6");
  option_parser_register(opp, "-trace_enabled", OPT_BOOL, &Trace::enabled,
                         "Turn on traces", "0");
  option_parser_register(opp, "-trace_components", OPT_CSTR, &Trace::config_str,
                         "comma seperated list of traces to enable. "
                         "Complete list found in trace_streams.tup. "
                         "Default none",
                         "none");
  option_parser_register(
      opp, "-trace_sampling_core", OPT_INT32, &Trace::sampling_core,
      "The core which is printed using CORE_DPRINTF. Default 0", "0");
  option_parser_register(opp, "-trace_sampling_memory_partition", OPT_INT32,
                         &Trace::sampling_memory_partition,
                         "The memory partition which is printed using "
                         "MEMPART_DPRINTF. Default -1 (i.e. all)",
                         "-1");
  option_parser_register(
      opp, "-timeshare_enabled", OPT_BOOL, &timeshare_enabled,
      "Enables timesharing the GPU between applications", "0");
  ptx_file_line_stats_options(opp);
}

/////////////////////////////////////////////////////////////////////////////

void increment_x_then_y_then_z(dim3& i, const dim3& bound) {
  i.x++;
  if (i.x >= bound.x) {
    i.x = 0;
    i.y++;
    if (i.y >= bound.y) {
      i.y = 0;
      if (i.z < bound.z)
        i.z++;
    }
  }
}

void gpgpu_sim::launch(kernel_info_t* kinfo) {
  unsigned cta_size = kinfo->threads_per_cta();
  if (cta_size > m_shader_config->n_thread_per_shader) {
    std::cerr
        << "Execution error: Shader kernel CTA (block) size is too large for microarch \
        config."
        << std::endl;
    std::cerr << "CTA size (x*y*z) = " << cta_size
              << ", max supported = " << m_shader_config->n_thread_per_shader
              << std::endl;
    std::cerr
        << "=> either change -gpgpu_shader argument in gpgpusim.config file or modify the \
        CUDA source to decrease the kernel block size."
        << std::endl;
    exit(EXIT_FAILURE);
  }
  unsigned n;
  for (n = 0; n < m_running_kernels.size(); n++) {
    std::cout << "GPGPU-sim: Launching Kernel " << kinfo->name()
              << " : n = " << n
              << ". Running kernel size = " << m_running_kernels.size()
              << std::endl;
    if ((NULL == m_running_kernels[n]) || m_running_kernels[n]->done()) {
      m_running_kernels[n] = kinfo;
      break;
    }
  }
  assert(n < m_running_kernels.size());
}

bool gpgpu_sim::can_start_kernel() {
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if ((NULL == m_running_kernels[n]) || m_running_kernels[n]->done())
      return true;
  }
  return false;
}

bool gpgpu_sim::get_more_cta_left() const {
  if (m_config.gpu_max_cta_opt != 0) {
    if (m_total_cta_launched >= m_config.gpu_max_cta_opt)
      return false;
  }
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if (m_running_kernels[n] && !m_running_kernels[n]->no_more_ctas_to_run())
      return true;
  }
  return false;
}

kernel_info_t* gpgpu_sim::select_kernel() {
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    unsigned idx =
        (n + m_last_issued_kernel + 1) % m_config.max_concurrent_kernel;
    if (m_running_kernels[idx] &&
        !m_running_kernels[idx]->no_more_ctas_to_run()) {
      m_last_issued_kernel = idx;
      // record this kernel for stat print if it is the first time this kernel
      // is selected for execution
      unsigned launch_uid = m_running_kernels[idx]->get_uid();
      if (std::find(m_executed_kernel_uids.begin(),
                    m_executed_kernel_uids.end(),
                    launch_uid) == m_executed_kernel_uids.end()) {
        m_executed_kernel_uids.push_back(launch_uid);
        m_executed_kernel_names.push_back(m_running_kernels[idx]->name());
      }

      return m_running_kernels[idx];
    }
  }
  return NULL;
}

unsigned gpgpu_sim::finished_kernel() {
  if (m_finished_kernel.empty())
    return 0;
  unsigned result = m_finished_kernel.front();
  m_finished_kernel.pop_front();
  return result;
}

void gpgpu_sim::set_kernel_done(kernel_info_t* kernel) {
  unsigned uid = kernel->get_uid();
  m_finished_kernel.push_back(uid);
  std::vector<kernel_info_t*>::iterator k;
  for (k = m_running_kernels.begin(); k != m_running_kernels.end(); k++) {
    if (*k == kernel) {
      *k = NULL;
      break;
    }
  }
  assert(k != m_running_kernels.end());
}

void set_ptx_warp_size(const struct core_config* warp_size);

gpgpu_sim::gpgpu_sim(const gpgpu_sim_config& config)
    : gpgpu_t(config, config.get_ptx_inst_debug_file()),
      m_config(config),
      m_shader_config(&config.m_shader_config),
      m_memory_config(&config.m_memory_config) {
  // set groups based on n_apps
  set_ptx_warp_size(m_shader_config);
  ptx_file_line_stats_create_exposed_latency_tracker(m_config.num_shader());

  std::cout << "Initializing shader stats" << std::endl;
  m_shader_stats = new shader_core_stats(m_shader_config);
  std::cout << "Initializing memory stats" << std::endl;
  m_memory_stats = new memory_stats_t(m_config.num_shader(), m_shader_config,
                                      m_memory_config);

  std::cout << "Initializing MMU" << std::endl;
  m_page_manager = g_mmu;

  // FIXME: Check this
  if (m_page_manager->need_init)
    m_page_manager->init2(m_memory_config);

  m_page_manager->set_stat(m_memory_stats);
  // call init next
}

int gpgpu_sim::shared_mem_size() const {
  return m_shader_config->gpgpu_shmem_size;
}

int gpgpu_sim::num_registers_per_core() const {
  return m_shader_config->gpgpu_shader_registers;
}

int gpgpu_sim::wrp_size() const {
  return m_shader_config->warp_size;
}

int gpgpu_sim::shader_clock() const {
  return m_config.core_freq / 1000;
}

void gpgpu_sim::set_prop(cudaDeviceProp* prop) {
  m_cuda_properties = prop;
}

const struct cudaDeviceProp* gpgpu_sim::get_prop() const {
  return m_cuda_properties;
}

enum divergence_support_t gpgpu_sim::simd_model() const {
  return m_shader_config->model;
}

void gpgpu_sim_config::init_clock_domains(void) {
  sscanf(gpgpu_clock_domains, "%lf:%lf:%lf:%lf", &core_freq, &icnt_freq,
         &l2_freq, &dram_freq);
  core_freq = core_freq MhZ;
  icnt_freq = icnt_freq MhZ;
  l2_freq = l2_freq MhZ;
  dram_freq = dram_freq MhZ;
  core_period = 1 / core_freq;
  icnt_period = 1 / icnt_freq;
  dram_period = 1 / dram_freq;
  l2_period = 1 / l2_freq;
  DEBUG_GPU << "clock freqs: " << core_freq << ":" << icnt_freq << ":" << l2_freq << ":" <<
      dram_freq << std::endl;
  DEBUG_GPU << "clock periods: " << core_period << ":" << icnt_period << ":" << l2_period <<
      ":" << dram_period << std::endl;
}

void gpgpu_sim::reinit_clock_domains(void) {
  core_time = 0;
  dram_time = 0;
  icnt_time = 0;
  l2_time = 0;
}

bool gpgpu_sim::active() {
  if (m_config.gpu_max_cycle_opt &&
      (gpu_tot_sim_cycle + gpu_sim_cycle) >= m_config.gpu_max_cycle_opt)
    return false;
  if (m_config.gpu_max_cta_opt &&
      (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt))
    return false;
  if (m_config.gpu_deadlock_detect && gpu_deadlock)
    return false;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    if (m_cluster[i]->get_not_completed() > 0)
      return true;
  ;
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
    if (m_memory_partition_unit[i]->busy() > 0)
      return true;
  ;
  if (icnt_busy())
    return true;
  if (get_more_cta_left())
    return true;
  return false;
}

void gpgpu_sim::init() {
  std::cout << "Initializing GPGPU-sim" << std::endl;
  // wait for all apps to be created
  while (App::get_created_app_count() != ConfigOptions::n_apps) {
    sleep(1);
  }
  m_memory_stats->init();
  std::cout << "Done initializing memory stats in GPGPU-sim. Initializing "
               "shader stats"
            << std::endl;
  m_shader_stats->init();

  average_pipeline_duty_cycle = (float*)malloc(sizeof(float));
  active_sms = (float*)malloc(sizeof(float));
  m_power_stats =
      new power_stat_t(m_shader_config, average_pipeline_duty_cycle, active_sms,
                       m_shader_stats, m_memory_config, m_memory_stats);

  gpu_sim_insn = 0;
  gpu_tot_sim_insn = 0;
  gpu_tot_issued_cta = 0;
  gpu_deadlock = false;
  max_insn_struck = false;  // important
  for (unsigned i = 0; i < m_config.num_shader(); i++) {
    gpu_sim_insn_per_core[i] = 0;
  }

  std::cout << "Done initializing shader stat. Initializing shared TLB"
            << std::endl;
  // Add a pointer to memory_partition unit so that tlb can insert DRAM
  // copy/zero commands
  m_shared_tlb =
      new tlb_tag_array(m_memory_config, m_shader_stats, m_page_manager, true,
                        m_memory_stats, m_memory_partition_unit);
  std::cout << "Done initializing shared TLB" << std::endl;

  m_cluster = new simt_core_cluster*[m_shader_config->n_simt_clusters];
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i] = new simt_core_cluster(
        this, i, m_shader_config, m_memory_config, m_shader_stats,
        m_memory_stats, m_page_manager, m_shared_tlb);

  m_memory_partition_unit =
      new memory_partition_unit*[m_memory_config->m_n_mem];
  m_memory_sub_partition =
      new memory_sub_partition*[m_memory_config->m_n_mem_sub_partition];
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
    m_memory_partition_unit[i] = new memory_partition_unit(
        i, m_memory_config, m_memory_stats, m_page_manager, m_shared_tlb);
    for (unsigned p = 0;
         p < m_memory_config->m_n_sub_partition_per_memory_channel; p++) {
      unsigned submpid =
          i * m_memory_config->m_n_sub_partition_per_memory_channel + p;
      m_memory_sub_partition[submpid] =
          m_memory_partition_unit[i]->get_sub_partition(p);
    }
  }

  icnt_wrapper_init();
  icnt_create(m_shader_config->n_simt_clusters,
              m_memory_config->m_n_mem_sub_partition);

  // FIXME: Check this
  if (m_page_manager->need_init)
    m_page_manager->set_ready();

  time_vector_create(NUM_MEM_REQ_STAT);
  DEBUG_GPU << "performance model initialization complete." << std::endl;

  m_running_kernels.resize(m_config.max_concurrent_kernel, NULL);
  m_last_issued_kernel = -1;  // 0
  m_last_cluster_issue = -1;  // 0
  *average_pipeline_duty_cycle = 0;
  *active_sms = 0;

  last_liveness_message_time = 0;
  // run a CUDA grid on the GPU microarchitecture simulator
  gpu_sim_cycle = 0;
  gpu_sim_insn = 0;
  last_gpu_sim_insn = 0;
  m_total_cta_launched = 0;
  count_tlp = 0;
  ConfigOptions::n_sms = m_config.num_shader();

  reinit_clock_domains();
  set_param_gpgpu_num_shaders(m_config.num_shader());
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i]->reinit();
  m_shader_stats->new_grid();
  // initialize the control-flow, memory access, memory latency logger
  if (m_config.g_visualizer_enabled) {
    create_thread_CFlogger(m_config.num_shader(),
                           m_shader_config->n_thread_per_shader, 0,
                           m_config.gpgpu_cflog_interval);
  }
  shader_CTA_count_create(m_config.num_shader(), m_config.gpgpu_cflog_interval);
  if (m_config.gpgpu_cflog_interval != 0) {
    insn_warp_occ_create(m_config.num_shader(), m_shader_config->warp_size);
    shader_warp_occ_create(m_config.num_shader(), m_shader_config->warp_size,
                           m_config.gpgpu_cflog_interval);
    shader_mem_acc_create(m_config.num_shader(), m_memory_config->m_n_mem, 4,
                          m_config.gpgpu_cflog_interval);
    shader_mem_lat_create(m_config.num_shader(), m_config.gpgpu_cflog_interval);
    shader_cache_access_create(m_config.num_shader(), 3,
                               m_config.gpgpu_cflog_interval);
    set_spill_interval(m_config.gpgpu_cflog_interval * 40);
  }

  if (g_network_mode)
    icnt_init();

// McPAT initialization function. Called on first launch of GPU
#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    // init_mcpat(m_config, m_gpgpusim_wrapper, m_config.gpu_stat_sample_freq,
    // gpu_tot_sim_insn, gpu_sim_insn);
  }
#endif
}

void gpgpu_sim::update_stats() {
  m_memory_stats->memlatstat_lat_pw();
  gpu_tot_sim_cycle += gpu_sim_cycle;
  gpu_tot_sim_insn += gpu_sim_insn;
}

void gpgpu_sim::print_stats() {
  ptx_file_line_stats_write_file();
  gpu_print_stat_file(std::cout);

  if (g_network_mode) {
    std::cout
        << "-----------------------Interconnect-DETAILS-----------------------"
        << std::endl;
    icnt_display_stats();
    icnt_display_overall_stats();
    std::cout << "-----------------------END-of-Interconnect-DETAILS-----------"
                 "------------"
              << std::endl;
  }
}

void gpgpu_sim::deadlock_check() {
  if (m_config.gpu_deadlock_detect && gpu_deadlock) {
    std::cout << "Deadlock detected" << std::endl;
    DEBUG_GPU << "DEADLOCK last writeback core " << gpu_sim_insn_last_update_sid <<
        " @ gpu_sim_cycle " << gpu_sim_insn_last_update << " (+ gpu_tot_sim_cycle "
        << gpu_tot_sim_cycle - gpu_sim_cycle << ") ("
        << gpu_sim_cycle - gpu_sim_insn_last_update << " cycles ago)"
        << std::endl;

    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      if (m_memory_partition_unit[i]->busy()) {
        DEBUG_GPU << "DEADLOCK memory partition " << i << " busy" << std::endl;
      }
    }
    if (icnt_busy()) {
      DEBUG_GPU << "DEADLOCK iterconnect contains traffic" << std::endl;
    }
    exit(EXIT_FAILURE);
  }
  if (m_config.gpu_max_insn_opt && max_insn_struck) {
    print_stats();
    std::cout << "MAX INSTRUCTIONS STRUCK" << std::endl;
    exit(EXIT_SUCCESS);
  }
}

void gpgpu_sim::app_cache_flush(appid_t appid) {
  std::vector<uint32_t> sms = App::get_app_sms(appid);
  for (std::vector<uint32_t>::iterator it = sms.begin(); it != sms.end();
       it++) {
    m_cluster[*it]->cache_flush();
  }
}

unsigned long long gpgpu_sim::get_gpu_insn_max() {
  return m_config.gpu_max_insn_opt;
}

/// printing the names and uids of a set of executed kernels (usually there is
/// only one)
std::string gpgpu_sim::executed_kernel_info_string() {
  std::stringstream statout;
  statout << "kernel_name = ";
  for (unsigned int k = 0; k < m_executed_kernel_names.size(); k++) {
    statout << m_executed_kernel_names[k] << " ";
  }
  statout << std::endl;
  statout << "kernel_launch_uid = ";
  for (unsigned int k = 0; k < m_executed_kernel_uids.size(); k++) {
    statout << m_executed_kernel_uids[k] << " ";
  }
  statout << std::endl;

  return statout.str();
}

void gpgpu_sim::set_cache_config(std::string kernel_name,
                                 FuncCache cacheConfig) {
  m_special_cache_config[kernel_name] = cacheConfig;
}

FuncCache gpgpu_sim::get_cache_config(std::string kernel_name) {
  for (std::map<std::string, FuncCache>::iterator iter =
           m_special_cache_config.begin();
       iter != m_special_cache_config.end(); iter++) {
    std::string kernel = iter->first;
    if (kernel_name.compare(kernel) == 0) {
      return iter->second;
    }
  }
  return (FuncCache)0;
}

bool gpgpu_sim::has_special_cache_config(std::string kernel_name) {
  for (std::map<std::string, FuncCache>::iterator iter =
           m_special_cache_config.begin();
       iter != m_special_cache_config.end(); iter++) {
    std::string kernel = iter->first;
    if (kernel_name.compare(kernel) == 0) {
      return true;
    }
  }
  return false;
}

void gpgpu_sim::set_cache_config(std::string kernel_name) {
  if (has_special_cache_config(kernel_name)) {
    change_cache_config(get_cache_config(kernel_name));
  } else {
    change_cache_config(FuncCachePreferNone);
  }
}

void gpgpu_sim::change_cache_config(FuncCache cache_config) {
  if (cache_config != m_shader_config->m_L1D_config.get_cache_status()) {
    std::cout << "FLUSH L1 Cache at configuration change between kernels"
              << std::endl;
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      m_cluster[i]->cache_flush();
    }
  }

  switch (cache_config) {
    case FuncCachePreferNone:
      m_shader_config->m_L1D_config.init(
          m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
      m_shader_config->gpgpu_shmem_size =
          m_shader_config->gpgpu_shmem_sizeDefault;
      break;
    case FuncCachePreferL1:
      if ((m_shader_config->m_L1D_config.m_config_stringPrefL1 == NULL) ||
          (m_shader_config->gpgpu_shmem_sizePrefL1 == (unsigned)-1)) {
        std::cout << "WARNING: missing Preferred L1 configuration" << std::endl;
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizeDefault;
      } else {
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_stringPrefL1,
            FuncCachePreferL1);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizePrefL1;
      }
      break;
    case FuncCachePreferShared:
      if ((m_shader_config->m_L1D_config.m_config_stringPrefShared == NULL) ||
          (m_shader_config->gpgpu_shmem_sizePrefShared == (unsigned)-1)) {
        std::cout << "WARNING: missing Preferred L1 configuration" << std::endl;
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizeDefault;
      } else {
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_stringPrefShared,
            FuncCachePreferShared);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizePrefShared;
      }
      break;
    default:
      break;
  }
}

void gpgpu_sim::clear_executed_kernel_info() {
  m_executed_kernel_names.clear();
  m_executed_kernel_uids.clear();
}

void gpgpu_sim::gpu_print_stat_file(std::ostream& out) {
  for (std::vector<App*>::iterator it = App::begin(); it != App::end(); it++) {
    out << "gpu_ipc_ " << (*it)->appid << " = "
        << (float)(*it)->gpu_sim_instruction_count /
               (*it)->gpu_total_simulator_cycles_stream
        << std::endl;
  }
  for (std::vector<App*>::iterator it = App::begin(); it != App::end(); it++) {
    out << "gpu_tot_sim_cycle_stream_" << (*it)->appid << " = "
        << (*it)->gpu_total_simulator_cycles_stream << std::endl;
  }
  for (std::vector<App*>::iterator it = App::begin(); it != App::end(); it++) {
    out << "gpu_sim_insn_" << (*it)->appid << " = "
        << (*it)->gpu_sim_instruction_count << std::endl;
  }

  out << "gpu_sim_cycle = " << gpu_sim_cycle << std::endl;
  out << "gpu_sim_insn = " << gpu_sim_insn << std::endl;
  out << "gpu_ipc = " << (float)gpu_sim_insn / gpu_sim_cycle << std::endl;
  out << "gpu_tot_sim_cycle = " << gpu_tot_sim_cycle + gpu_sim_cycle
      << std::endl;
  out << "gpu_tot_sim_insn = " << gpu_tot_sim_insn + gpu_sim_insn << std::endl;
  out << "gpu_tot_ipc = "
      << (float)(gpu_tot_sim_insn + gpu_sim_insn) /
             (gpu_tot_sim_cycle + gpu_sim_cycle)
      << std::endl;
  out << "gpu_tot_issued_cta = " << gpu_tot_issued_cta << std::endl;

  // performance counter for stalls due to congestion.
  out << "gpu_stall_dramfull = " << gpu_stall_dramfull << std::endl;
  out << "gpu_stall_icnt2sh = " << gpu_stall_icnt2sh << std::endl;

  time_t curr_time;
  time(&curr_time);
  unsigned long long elapsed_time = MAX(curr_time - g_simulation_starttime, 1);
  out << "gpu_total_sim_rate = "
      << (unsigned)((gpu_tot_sim_insn + gpu_sim_insn) / elapsed_time)
      << std::endl;

  shader_print_cache_stats(out);
  cache_stats core_cache_stats;
  core_cache_stats.clear();
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    m_cluster[i]->get_cache_stats(core_cache_stats);
  }
  out << std::endl << "Total_core_cache_stats:" << std::endl;
  core_cache_stats.print_stats(out, "Total_core_cache_stats_breakdown");
  shader_print_scheduler_stat(out, false);

  m_shader_stats->print(out);
#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    // m_gpgpusim_wrapper->print_power_kernel_stats(gpu_sim_cycle,
    // gpu_tot_sim_cycle, gpu_tot_sim_insn + gpu_sim_insn, kernel_info_str, true
    // );
    // mcpat_reset_perf_count(m_gpgpusim_wrapper);
  }
#endif

  // performance counter that are not local to one shader
  m_memory_stats->memlatstat_print_file(m_memory_config->m_n_mem,
                                        m_memory_config->nbk, out);
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
    out << m_memory_partition_unit[i] << std::endl;

  // L2 cache stats
  if (!m_memory_config->m_L2_config.disabled()) {
    cache_stats l2_stats;
    struct cache_sub_stats l2_css;
    struct cache_sub_stats total_l2_css;
    l2_stats.clear();
    l2_css.clear();
    total_l2_css.clear();

    out << std::endl
        << "-----------------------L2 cache stats-----------------------"
        << std::endl;
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      m_memory_sub_partition[i]->accumulate_L2cache_stats(l2_stats);
      m_memory_sub_partition[i]->get_L2cache_sub_stats(l2_css);

      out << "L2_cache_bank[" << i << "]: Access = " << l2_css.accesses
          << ", Miss = " << l2_css.misses
          << ", Miss_rate = " << (double)l2_css.misses / l2_css.accesses
          << ", Pending_hits = " << l2_css.pending_hits
          << ", Reservation_fails = " << l2_css.res_fails << std::endl;
      total_l2_css += l2_css;
    }
    if (!m_memory_config->m_L2_config.disabled() &&
        m_memory_config->m_L2_config.get_num_lines()) {
      L2c_print_cache_stat(out);
      out << "L2_total_cache_accesses = " << total_l2_css.accesses << std::endl;
      out << "L2_total_cache_misses = " << total_l2_css.misses << std::endl;
      if (total_l2_css.accesses > 0)
        out << "L2_total_cache_miss_rate = "
            << (double)total_l2_css.misses / (double)total_l2_css.accesses
            << std::endl;
      out << "L2_total_cache_pending_hits = " << total_l2_css.pending_hits
          << std::endl;
      out << "L2_total_cache_reservation_fails = " << total_l2_css.res_fails
          << std::endl;
      out << "L2_total_cache_breakdown:" << std::endl;
      l2_stats.print_stats(out, "L2_cache_stats_breakdown");
      total_l2_css.print_port_stats(out, "L2_cache");
    }
  }

  if (m_config.gpgpu_cflog_interval != 0) {
    spill_log_to_file(out, 1, gpu_sim_cycle);
    insn_warp_occ_print(out);
  }
  if (gpgpu_ptx_instruction_classification) {
    StatDisp(g_inst_classification_stat[g_ptx_kernel_count]);
    StatDisp(g_inst_op_classification_stat[g_ptx_kernel_count]);
  }

#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    // m_gpgpusim_wrapper->detect_print_steady_state(1,gpu_tot_sim_insn+gpu_sim_insn);
  }
#endif
  // Interconnect power stat print
  long total_simt_to_mem = 0;
  long total_mem_to_simt = 0;
  long temp_stm = 0;
  long temp_mts = 0;
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    m_cluster[i]->get_icnt_stats(temp_stm, temp_mts);
    total_simt_to_mem += temp_stm;
    total_mem_to_simt += temp_mts;
  }
  out << std::endl
      << "icnt_total_pkts_mem_to_simt=" << total_mem_to_simt << std::endl;
  out << "icnt_total_pkts_simt_to_mem=" << total_simt_to_mem << std::endl;

  // time_vector_print();
  // fflush(stdout);

  // clear_executed_kernel_info();
}

// performance counter that are not local to one shader
unsigned gpgpu_sim::threads_per_core() const {
  return m_shader_config->n_thread_per_shader;
}

void shader_core_ctx::mem_instruction_stats(const warp_inst_t& inst) {
  unsigned active_count = inst.active_count();
  // this breaks some encapsulation: the is_[space] functions, if you change
  // those, change this.
  switch (inst.space.get_type()) {
    case undefined_space:
    case reg_space:
      break;
    case shared_space:
      m_stats->gpgpu_n_shmem_insn += active_count;
      break;
    case const_space:
      m_stats->gpgpu_n_const_insn += active_count;
      break;
    case param_space_kernel:
    case param_space_local:
      m_stats->gpgpu_n_param_insn += active_count;
      break;
    case tex_space:
      m_stats->gpgpu_n_tex_insn += active_count;
      break;
    case global_space:
    case local_space:
      if (inst.is_store())
        m_stats->gpgpu_n_store_insn += active_count;
      else
        m_stats->gpgpu_n_load_insn += active_count;
      break;
    default:
      assert(false && "Invalid instruction space type");
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Launches a cooperative thread array (CTA).
 *
 * @param kernel
 *    object that tells us which kernel to ask for a CTA from
 */

void shader_core_ctx::issue_block2core(kernel_info_t& kernel) {
  set_max_cta(kernel);

  unsigned kernel_id = kernel.get_uid();
  unsigned stream_id = kernel.get_stream_id();

  // find a free CTA context
  unsigned free_cta_hw_id = (unsigned)-1;
  for (unsigned i = 0; i < kernel_max_cta_per_shader; i++) {
    if (m_cta_status[i] == 0) {
      free_cta_hw_id = i;
      break;
    }
  }
  assert(free_cta_hw_id != (unsigned)-1);

  // determine hardware threads and warps that will be used for this CTA
  int cta_size = kernel.threads_per_cta();

  // hw warp id = hw thread id mod warp size, so we need to find a range
  // of hardware thread ids corresponding to an integral number of hardware
  // thread ids
  int padded_cta_size = cta_size;
  if (cta_size % m_config->warp_size)
    padded_cta_size =
        ((cta_size / m_config->warp_size) + 1) * (m_config->warp_size);
  unsigned start_thread = free_cta_hw_id * padded_cta_size;
  unsigned end_thread = start_thread + cta_size;

  // reset the microarchitecture state of the selected hardware thread and warp
  // contexts
  reinit(start_thread, end_thread, false);

  // initalize scalar threads and determine which hardware warps they are
  // allocated to
  // bind functional simulation state of threads to hardware resources
  // (simulation)
  warp_set_t warps;
  unsigned nthreads_in_block = 0;
  for (unsigned i = start_thread; i < end_thread; i++) {
    m_threadState[i].m_cta_id = free_cta_hw_id;
    unsigned warp_id = i / m_config->warp_size;
    nthreads_in_block += ptx_sim_init_thread(
        kernel, &m_thread[i], m_sid, i, cta_size - (i - start_thread),
        m_config->n_thread_per_shader, this, free_cta_hw_id, warp_id,
        m_cluster->get_gpu());
    m_threadState[i].m_active = true;
    warps.set(warp_id);
  }
  assert(nthreads_in_block > 0 &&
         nthreads_in_block <= m_config->n_thread_per_shader);  // should be at
                                                               // least one, but
                                                               // less than max
  m_cta_status[free_cta_hw_id] = nthreads_in_block;

  // now that we know which warps are used in this CTA, we can allocate
  // resources for use in CTA-wide barrier operations
  m_barriers.allocate_barrier(free_cta_hw_id, warps);

  // initialize the SIMT stacks and fetch hardware
  init_warps(free_cta_hw_id, start_thread, end_thread);
  m_n_active_cta++;

  shader_CTA_count_log(m_sid, 1);
  DEBUG_GPU << "Shader:" << m_sid << ", cta:" << free_cta_hw_id
            << " initialized @(" << gpu_sim_cycle << ", " << gpu_tot_sim_cycle
            << "), ACTIVE=" << m_n_active_cta << ", KERNEL=" << kernel_id
            << ", STREAM=" << stream_id << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////

void dram_t::dram_log(int task) {
  if (task == SAMPLELOG) {
    StatAddSample(mrqq_Dist, que_length());
  } else if (task == DUMPLOG) {
    std::cout << "Queue Length DRAM[" << id << "]" << std::endl;
    StatDisp(mrqq_Dist);
  }
}

// Find next clock domain and increment its time
int gpgpu_sim::next_clock_domain(void) {
  double smallest = min3(core_time, icnt_time, dram_time);
  int mask = 0x00;
  if (l2_time <= smallest) {
    smallest = l2_time;
    mask |= L2;
    l2_time += m_config.l2_period;
  }
  if (icnt_time <= smallest) {
    mask |= ICNT;
    icnt_time += m_config.icnt_period;
  }
  if (dram_time <= smallest) {
    mask |= DRAM;
    dram_time += m_config.dram_period;
  }
  if (core_time <= smallest) {
    mask |= CORE;
    core_time += m_config.core_period;
  }
  return mask;
}

void gpgpu_sim::issue_block2core() {
  unsigned last_issued = m_last_cluster_issue;
  unsigned clusters_issued;
  if (m_memory_config->gpu_char == 0) {
    clusters_issued = m_shader_config->n_simt_clusters;
  } else {
    clusters_issued = m_shader_config->n_simt_clusters / 2;
  }
  for (unsigned i = 0; i < clusters_issued; i++) {
    unsigned idx = (i + last_issued + 1) % clusters_issued;
    ;
    unsigned num = m_cluster[idx]->issue_block2core();
    if (num) {
      m_last_cluster_issue = idx;
      m_total_cta_launched += num;
    }
  }
}

unsigned long long g_single_step = 0;  // set this in gdb to single step the pipeline

void gpgpu_sim::cycle() {
  static Schedules::context_switch_state context_switching = Schedules::NORMAL;
  static Schedule_Assignment new_schedule;
  static std::set<memory_partition_unit*> dumping_partitions;
  static uint64_t last_schedule = 0;
  static Schedule_Assignment (*scheduler)(void) = &Schedules::assign_rr_schedule;
  static std::mutex schedule_mutex;

  if (gpu_sim_cycle == 0) {
    // Initialize the scheduler. This could be in a different function, called before cycling begins
    if (m_config.timeshare_enabled) {
      new_schedule = scheduler();
    } else {
      // SM partitioning only when timeshare is not enabled
      new_schedule = Schedules::assign_priority_schedule();
    }
    App::set_app_sms(new_schedule.new_assignment);
    // set the register files for each app
    std::vector<uint32_t> sms(ConfigOptions::n_sms);
    std::iota(sms.begin(), sms.end(), 0);
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i) {
      m_cluster[i]->context_switch(sms);
    }
  }

  if (schedule_mutex.try_lock()) {
    // scheduling needs to run in one and only one thread
    switch (context_switching) {
      case Schedules::NORMAL:
        // normal execution - wait to trigger context switch
        if (m_config.timeshare_enabled && // do not context switch if it is disabled
            (gpu_sim_cycle - last_schedule >= ConfigOptions::schedule_cycles)) {
          new_schedule = scheduler();
          if (new_schedule.evicted.size()) {
            context_switching = Schedules::SM_DISABLE;
            DEBUG_SCHED << "to SM_DISABLE at cycle " << gpu_sim_cycle << std::endl;
          }
        }
        break;
      case Schedules::SM_DISABLE:
        // Disable instruction fetch on sms that will be context switched
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
          m_cluster[i]->disable_inst_fetch(new_schedule.evicted_sms);
        }
        context_switching = Schedules::PIPE_DRAIN;
        DEBUG_SCHED << "to PIPE_DRAIN at cycle " << gpu_sim_cycle << std::endl;
        break;
      case Schedules::PIPE_DRAIN:
        {
          // wait for SMs to finish executing in-flight instructions
          unsigned in_flight = 0;
          for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
            if (in_flight += m_cluster[i]->instructions_in_flight_for_sms(
                  new_schedule.evicted_sms)) {
              // if any instruction is in flight, we must continue waiting
              //in_flight = true;
              //break;
            }
          }
          if (in_flight) {
            break;
          } else {
            // no instructions are in flight
            context_switching = Schedules::REG_DUMP;
            DEBUG_SCHED << "to REG_DUMP at cycle " << gpu_sim_cycle << std::endl;
            // no break, continue to next case on same cycle
          }
        }
      case Schedules::REG_DUMP:
        // Flush register state for evicted sms
        for (std::map<appid_t, std::vector<uint32_t>>::const_iterator it =
            new_schedule.evicted.cbegin();
            it != new_schedule.evicted.cend(); it++) {
          size_t register_size = num_registers_per_core() * 8; // 8 bytes per register
          unsigned line_size = m_shader_config->m_L1D_config.get_line_sz();
          new_addr_type dump_to = 0x800 + register_size * App::get_app(it->first)->addr_offset; // some high address
          for (std::vector<uint32_t>::const_iterator sm = it->second.cbegin();
              sm != it->second.cend(); sm++) {
            for (new_addr_type offset = 0; offset < register_size; offset += line_size) {
              mem_fetch* dump = new mem_fetch(m_memory_config, dump_to + offset, line_size, *sm,
                  WRITE_REQUEST);
              dumping_partitions.insert(m_memory_partition_unit[dump->get_partition_addr()]);
              m_memory_partition_unit[dump->get_partition_addr()]->dram_push(dump);
            }
          }
        }
        context_switching = Schedules::REG_DUMP_DRAIN;
        DEBUG_SCHED << "to REG_DUMP_DRAIN at cycle " << gpu_sim_cycle << std::endl;
        break;
      case Schedules::REG_DUMP_DRAIN:
        {
          // wait for waitlist to empty before proceeding to next phase
          bool draining = false;
          for (std::set<memory_partition_unit*>::const_iterator it = dumping_partitions.cbegin();
              it != dumping_partitions.cend(); it++) {
            if ((*it)->dram_queue_size()) {
              // memory operations are still one the queue
              draining = true;
              break;
            }
          }
          if (!draining) {
            break;
          } else {
            // no memory operations are on the queue
            dumping_partitions.clear();
            context_switching = Schedules::CACHE_DUMP;
            DEBUG_SCHED << "to CACHE_DUMP at cycle " << gpu_sim_cycle << std::endl;
            // no break, continue to next case on same cycle
          }
        }
      case Schedules::CACHE_DUMP:
        // flush the TLB for apps that are evicted
        if (m_memory_config->tlb_enable) {
          for (std::map<appid_t, std::vector<uint32_t>>::const_iterator it =
              new_schedule.evicted.cbegin();
              it != new_schedule.evicted.cend(); it++) {
            m_shared_tlb->flush(it->first);
          }
        } else {
          // flush the L1 caches
          // Should we only do this for SMS that had apps that were evicted?
          // This flushes all L1 caches
          // TODO verify that cache_flush also writes back dirty lines
          for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
            m_cluster[i]->cache_flush();
          }
        }
        context_switching = Schedules::CACHE_DUMP_DRAIN;
        DEBUG_SCHED << "to CACHE_DUMP_DRAIN at cycle " << gpu_sim_cycle << std::endl;
        break;
      case Schedules::CACHE_DUMP_DRAIN:
        // TODO figure this out
        context_switching = Schedules::REG_LOAD;
        // no break, continue to next case on same cycle
      case Schedules::REG_LOAD:
        // Load register state
        // (Simplifying assumption, if evicted, the newly scheduled job will need to load state)
        for (std::map<appid_t, std::vector<uint32_t>>::const_iterator it =
            new_schedule.evicted.cbegin();
            it != new_schedule.evicted.cend(); it++) {
          for (std::vector<uint32_t>::const_iterator sm = it->second.cbegin();
              sm != it->second.cend(); sm++) {
            size_t register_size = num_registers_per_core() * 8; // 8 bytes per register
            unsigned line_size = m_shader_config->m_L1D_config.get_line_sz();
            new_addr_type read_from = 0x800 + register_size *
                App::get_app(App::get_app_id_from_sm(*sm))->addr_offset; // some high address
            for (new_addr_type offset = 0; offset < register_size; offset += line_size) {
              mem_fetch* load = new mem_fetch(m_memory_config, read_from + offset, line_size, *sm,
                  READ_REQUEST);
              m_memory_partition_unit[load->get_partition_addr()]->dram_push(load);
            }
          }
        }
        context_switching = Schedules::SM_ENABLE;
        DEBUG_SCHED << "to SM_ENABLE at cycle " << gpu_sim_cycle << std::endl;
        break;
      case Schedules::SM_ENABLE:
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
          m_cluster[i]->context_switch(new_schedule.evicted_sms);
          m_cluster[i]->enable_inst_fetch(new_schedule.evicted_sms);
        }
        App::set_app_sms(new_schedule.new_assignment);
        context_switching = Schedules::NORMAL;
        DEBUG_SCHED << "to NORMAL at cycle " << gpu_sim_cycle << std::endl;
        last_schedule = gpu_sim_cycle;
        break;
      default:
        assert(false);
    }
    schedule_mutex.unlock();
  }

  int clock_mask = next_clock_domain();

  if (clock_mask & CORE) {
    // shader core loading (pop from ICNT into core) follows CORE clock
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
      m_cluster[i]->icnt_cycle();
  }
  if (clock_mask & ICNT) {
    // pop from memory controller to interconnect
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      mem_fetch* mf = m_memory_sub_partition[i]->top();
      if (mf) {
        unsigned response_size =
            mf->get_is_write() ? mf->get_ctrl_size() : mf->size();
        if (::icnt_has_buffer(m_shader_config->mem2device(i), response_size)) {
          if (!mf->get_is_write())
            mf->set_return_timestamp(gpu_sim_cycle + gpu_tot_sim_cycle);
          mf->set_status(IN_ICNT_TO_SHADER, gpu_sim_cycle + gpu_tot_sim_cycle);
          ::icnt_push(m_shader_config->mem2device(i), mf->get_tpc(), mf,
                      response_size);
          m_memory_sub_partition[i]->pop();
        } else {
          gpu_stall_icnt2sh++;
        }
      } else {
        m_memory_sub_partition[i]->pop();
      }
    }
  }

  if (clock_mask & DRAM) {
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      m_memory_partition_unit[i]
          ->dram_cycle();  // Issue the dram command (scheduler + delay model)
      // Update performance counters for DRAM
      m_memory_partition_unit[i]->set_dram_power_stats(
          m_power_stats->pwr_mem_stat->n_cmd[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_activity[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_nop[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_act[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_pre[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_rd[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_wr[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_req[CURRENT_STAT_IDX][i]);
    }
  }

  // L2 operations follow L2 clock domain
  if (clock_mask & L2) {
    m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].clear();
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      // move memory request from interconnect into memory partition (if not
      // backed up)
      // Note:This needs to be called in DRAM clock domain if there is no L2
      // cache in the system
      if (m_memory_sub_partition[i]->full()) {
        gpu_stall_dramfull++;
      } else {
        mem_fetch* mf = (mem_fetch*)icnt_pop(m_shader_config->mem2device(i));
        m_memory_sub_partition[i]->push(mf, gpu_sim_cycle + gpu_tot_sim_cycle);
      }
      m_memory_sub_partition[i]->cache_cycle(gpu_sim_cycle + gpu_tot_sim_cycle);
      m_memory_sub_partition[i]->accumulate_L2cache_stats(
          m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX]);
    }
  }

  if (clock_mask & ICNT) {
    icnt_transfer();
  }

  if (clock_mask & CORE) {
    // L1 cache + shader core pipeline stages
    m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].clear();
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      if (m_cluster[i]->get_not_completed() || get_more_cta_left()) {
        m_cluster[i]->core_cycle();
        *active_sms += m_cluster[i]->get_n_active_sms();
      }
      // Update core icnt/cache stats for GPUWattch
      m_cluster[i]->get_icnt_stats(
          m_power_stats->pwr_mem_stat->n_simt_to_mem[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_mem_to_simt[CURRENT_STAT_IDX][i]);
      m_cluster[i]->get_cache_stats(
          m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX]);
    }
    float temp = 0;

    for (unsigned i = 0; i < m_shader_config->num_shader(); i++) {
      temp += m_shader_stats->m_pipeline_duty_cycle[i];
    }
    temp = temp / m_shader_config->num_shader();
    *average_pipeline_duty_cycle = ((*average_pipeline_duty_cycle) + temp);
    // cout<<"Average pipeline duty cycle:
    // "<<*average_pipeline_duty_cycle<<endl;

    if (g_single_step &&
        ((gpu_sim_cycle + gpu_tot_sim_cycle) >= g_single_step)) {
      asm("int $03");
    }
    gpu_sim_cycle++;

    my_active_sms = 0;
    for (unsigned i = 0; i < ConfigOptions::n_sms; i++) {
      if (m_cluster[i]->get_n_active_sms()) {
        my_active_sms++;
      }
    }
    for (std::vector<App*>::iterator it = App::begin(); it != App::end();
         it++) {
      // get all SMs assigned to an app
      const std::vector<uint32_t> app_sms = App::get_app_sms((*it)->appid);
      for (std::vector<uint32_t>::const_iterator sm = app_sms.cbegin();
           sm != app_sms.cend(); sm++) {
        // If any SM in m_cluster was active, increment total cycles.
        if (m_cluster[*sm]->get_n_active_sms() > 0) {
          (*it)->gpu_total_simulator_cycles_stream++;
          break;
        }
      }
    }

    if (DEBUG_ENABLE)
      gpgpu_debug();

// McPAT main cycle (interface with McPAT)
#ifdef GPGPUSIM_POWER_MODEL
    if (m_config.g_power_simulation_enabled) {
      // mcpat_cycle(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper,
      // m_power_stats, m_config.gpu_stat_sample_freq, gpu_tot_sim_cycle,
      // gpu_sim_cycle, gpu_tot_sim_insn, gpu_sim_insn);
    }
#endif

    issue_block2core();

    // Depending on configuration, flush the caches once all of threads are
    // completed.
    int all_threads_complete = 1;
    if (m_config.gpgpu_flush_l1_cache) {
      for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
        if (m_cluster[i]->get_not_completed() == 0)
          m_cluster[i]->cache_flush();
        else
          all_threads_complete = 0;
      }
    }

    if (m_config.gpgpu_flush_l2_cache) {
      if (!m_config.gpgpu_flush_l1_cache) {
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
          if (m_cluster[i]->get_not_completed() != 0) {
            all_threads_complete = 0;
            break;
          }
        }
      }

      if (all_threads_complete && !m_memory_config->m_L2_config.disabled()) {
        std::cout << "Flushed L2 caches..." << std::endl;
        if (m_memory_config->m_L2_config.get_num_lines()) {
          int dlc = 0;
          for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
            dlc = m_memory_sub_partition[i]->flushL2();
            assert(dlc == 0);  // need to model actual writes to DRAM here
            std::cout << "Dirty lines flushed from L2 " << i << " is " << dlc
                      << std::endl;
          }
        }
      }
    }

    if (!(gpu_sim_cycle % m_config.gpu_stat_sample_freq)) {
      time_t curr_time;
      time(&curr_time);
      time_t elapsed_time = curr_time - g_simulation_starttime;
      if ((elapsed_time - last_liveness_message_time) >=
          m_config.liveness_message_freq) {
        char elapsed_str[32];
        std::strftime(elapsed_str, 32, "%H:%M:%S", localtime(&elapsed_time));
        DEBUG_GPU << "cycles simulated: "
                  << gpu_tot_sim_cycle + gpu_sim_cycle
                  << " inst.: " << gpu_tot_sim_cycle + gpu_sim_insn
                  << " (ipc=" << (double)gpu_sim_insn / gpu_sim_cycle
                  << ") sim_rate="
                  << (unsigned)((gpu_tot_sim_insn + gpu_sim_insn) /
                                elapsed_time)
                  << " (inst/sec) elapsed = " << elapsed_str
                  << " current time = " << ctime(&curr_time) << std::endl;

        for (std::vector<App*>::iterator it = App::begin(); it != App::end();
             it++) {
          std::cout << "App: " << (*it)->appid << ", instructions: " <<
              (*it)->gpu_sim_instruction_count << ", cycles: " <<
              (*it)->gpu_total_simulator_cycles_stream << ", ipc: " <<
              (double) (*it)->gpu_sim_instruction_count /
              (*it)->gpu_total_simulator_cycles_stream << std::endl;
        }
        last_liveness_message_time = elapsed_time;
      }
      visualizer_printstat();
      m_memory_stats->memlatstat_lat_pw();
      if (m_config.gpgpu_runtime_stat &&
          (m_config.gpu_runtime_stat_flag != 0)) {
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_BW_STAT) {
          for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
            m_memory_partition_unit[i]->print_stat(std::cout);
          std::cout << "maxmrqlatency = " << m_memory_stats->max_mrq_latency
                    << std::endl;
          std::cout << "maxmflatency = " << m_memory_stats->max_mf_latency
                    << std::endl;
          std::cout << "high_prio_queue_drain_reset = "
                    << m_memory_stats->drain_reset << std::endl;
          std::cout << "average_combo_count = "
                    << m_memory_stats->total_combo /
                           m_memory_config->max_DRAM_high_prio_combo
                    << std::endl;
          std::cout << "sched_from_high_prio = "
                    << m_memory_stats->sched_from_high_prio
                    << ", DRAM_high_prio = " << m_memory_stats->DRAM_high_prio
                    << std::endl;
        }
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_L1MISS)
          shader_print_l1_miss_stat(std::cout);
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SCHED)
          shader_print_scheduler_stat(std::cout, false);
      }
    }

    if (!(gpu_sim_cycle % 2000000)) {
      // deadlock detection
      if (m_config.gpu_deadlock_detect && gpu_sim_insn == last_gpu_sim_insn) {
        gpu_deadlock = true;
      } else {
        last_gpu_sim_insn = gpu_sim_insn;
      }
    }

    if (m_config.gpu_max_insn_opt &&
        std::all_of(App::cbegin(), App::cend(), [&](App* app) {
          return app->gpu_sim_instruction_count >=
                 this->m_config.gpu_max_insn_opt;
        })) {
      max_insn_struck = true;
      print_stats();

      std::cout << "statistics when all apps completed MAX instructions"
                << std::endl;
      std::cout << "-------------------------------------------------"
                << std::endl;
      exit(EXIT_SUCCESS);
    }
    try_snap_shot(gpu_sim_cycle);
    spill_log_to_file(std::cout, 0, gpu_sim_cycle);
  }
}

void shader_core_ctx::dump_warp_state(std::ostream& out) const {
  out << std::endl;
  out << "per warp functional simulation status:" << std::endl;
  for (unsigned w = 0; w < m_config->max_warps_per_shader; w++)
    m_warp[w].print(out);
}

void gpgpu_sim::dump_pipeline(int mask, int s, int m) const {
  /*
     You may want to use this function while running GPGPU-Sim in gdb.
     One way to do that is add the following to your .gdbinit file:

     define dp
     call g_the_gpu.dump_pipeline_impl((0x40|0x4|0x1),$arg0,0)
     end

     Then, typing "dp 3" will show the contents of the pipeline for shader core
     3.
   */

  std::cout << "Dumping pipeline state..." << std::endl;
  if (!mask)
    mask = 0xFFFFFFFF;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    if (s != -1) {
      i = s;
    }
    if (mask & 1)
      m_cluster[m_shader_config->sid_to_cluster(i)]->display_pipeline(
          i, std::cout, 1, mask & 0x2E);
    if (s != -1) {
      break;
    }
  }
  if (mask & 0x10000) {
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      if (m != -1) {
        i = m;
      }
      std::cout << "DRAM / memory controller " << i << ":" << std::endl;
      if (mask & 0x100000)
        m_memory_partition_unit[i]->print_stat(std::cout);
      if (mask & 0x1000000)
        m_memory_partition_unit[i]->visualize();
      if (mask & 0x10000000)
        std::cout << m_memory_partition_unit[i];
      if (m != -1) {
        break;
      }
    }
  }
  fflush(stdout);
}

const struct shader_core_config* gpgpu_sim::getShaderCoreConfig() {
  return m_shader_config;
}

const struct memory_config* gpgpu_sim::getMemoryConfig() {
  return m_memory_config;
}

simt_core_cluster* gpgpu_sim::getSIMTCluster() {
  return *m_cluster;
}
