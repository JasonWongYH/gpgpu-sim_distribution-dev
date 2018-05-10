// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
// George L. Yuan
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

#include "mem_latency_stat.h"
#include "../abstract_hardware_model.h"
#include "../cuda-sim/ptx-stats.h"
#include "dram.h"
#include "gpu-cache.h"
#include "gpu-misc.h"
#include "gpu-sim.h"
#include "mem_fetch.h"
#include "shader.h"
#include "stat-tool.h"
#include "visualizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

memory_stats_t::memory_stats_t(unsigned n_shader,
                               const struct shader_core_config* shader_config,
                               const struct memory_config* mem_config) {
  assert(mem_config->m_valid);
  assert(shader_config->m_valid);

  concurrent_row_access =
      (uint64_t**)calloc(mem_config->m_n_mem, sizeof(uint64_t*));
  num_activates = (uint64_t**)calloc(mem_config->m_n_mem, sizeof(uint64_t*));
  num_activates_w = (uint64_t**)calloc(mem_config->m_n_mem, sizeof(uint64_t*));
  row_access = (uint64_t**)calloc(mem_config->m_n_mem, sizeof(uint64_t*));
  row_access_w = (uint64_t**)calloc(mem_config->m_n_mem, sizeof(uint64_t*));
  max_conc_access2samerow =
      (uint64_t**)calloc(mem_config->m_n_mem, sizeof(uint64_t*));
  max_servicetime2samerow =
      (uint64_t**)calloc(mem_config->m_n_mem, sizeof(uint64_t*));

  for (unsigned i = 0; i < mem_config->m_n_mem; i++) {
    concurrent_row_access[i] =
        (uint64_t*)calloc(mem_config->nbk, sizeof(uint64_t));
    num_activates[i] = (uint64_t*)calloc(mem_config->nbk, sizeof(uint64_t));
    row_access[i] = (uint64_t*)calloc(mem_config->nbk, sizeof(uint64_t));
    row_access_w[i] = (uint64_t*)calloc(mem_config->nbk, sizeof(uint64_t));
    num_activates_w[i] = (uint64_t*)calloc(mem_config->nbk, sizeof(uint64_t));
    max_conc_access2samerow[i] =
        (uint64_t*)calloc(mem_config->nbk, sizeof(uint64_t));
    max_servicetime2samerow[i] =
        (uint64_t*)calloc(mem_config->nbk, sizeof(uint64_t));
  }

  high_prio_queue_count = 0;
  dram_app_switch = 0;

  tlb_bypassed = 0;
  max_bloat = 0;
  coalesced_tried = 0;
  coalesced_succeed = 0;
  coalesced_noinval_succeed = 0;
  coalesced_fail = 0;
  num_coalesce = 0;
  pt_space_size = 0;

  for (int i = 0; i < 10; i++) {
    tlb_bypassed_level[i] = 0;
    tlb_level_accesses[i] = 0;
    tlb_level_hits[i] = 0;
    tlb_level_misses[i] = 0;
    tlb_level_fails[i] = 0;
  }
  for (int i = 0; i < 200; i++) {
    tlb_bypassed_core[i] = 0;
    TLBL1_sharer_avg[i] = 0.0;
    TLBL1_total_unique_addr[i] = 0;
    TLBL1_sharer_var[i] = 0.0;
    TLBL1_sharer_max[i] = 0;
    TLBL1_sharer_min[i] = 0;
    TLB_L1_flush_stalled[i] = 0;
  }

  TLBL2_sharer_avg = 0.0;
  TLBL2_total_unique_addr = 0;
  TLBL2_sharer_var = 0.0;
  TLBL2_sharer_max = 0;
  TLBL2_sharer_min = 0;
  TLB_L2_flush_stalled = 0;
  TLB_bypass_cache_flush_stalled = 0;

  m_n_shader = n_shader;
  m_memory_config = mem_config;
  total_n_access = 0;
  total_n_reads = 0;
  total_n_writes = 0;
  max_mrq_latency = 0;
  max_dq_latency = 0;
  max_mf_latency = 0;
  tlb_max_mf_latency = 0;
  max_icnt2mem_latency = 0;
  max_icnt2sh_latency = 0;
  memset(mrq_lat_table, 0, sizeof(unsigned) * 32);
  memset(dq_lat_table, 0, sizeof(unsigned) * 32);
  memset(mf_lat_table, 0, sizeof(unsigned) * 32);
  memset(icnt2mem_lat_table, 0, sizeof(unsigned) * 24);
  memset(icnt2sh_lat_table, 0, sizeof(unsigned) * 24);
  memset(mf_lat_pw_table, 0, sizeof(unsigned) * 32);

  DRAM_normal_prio = 0;
  DRAM_high_prio = 0;
  DRAM_special_prio = 0;
  sched_from_normal_prio = 0;
  sched_from_high_prio = 0;
  sched_from_special_prio = 0;
  drain_reset = 0;
  total_combo = 0;

  max_warps =
      n_shader *
      (shader_config->n_thread_per_shader / shader_config->warp_size + 1);

  totalbankreads = (uint64_t**)calloc(mem_config->m_n_mem, sizeof(uint64_t*));
  totalbankblocked = (uint64_t**)calloc(mem_config->m_n_mem, sizeof(uint64_t*));
  totalbankwrites = (uint64_t**)calloc(mem_config->m_n_mem, sizeof(uint64_t*));
  totalbankaccesses =
      (uint64_t**)calloc(mem_config->m_n_mem, sizeof(uint64_t*));
  mf_total_lat_table =
      (uint64_t**)calloc(mem_config->m_n_mem, sizeof(uint64_t*));
  mf_max_lat_table = (uint64_t**)calloc(mem_config->m_n_mem, sizeof(uint64_t*));
  bankreads = (uint64_t***)calloc(n_shader, sizeof(uint64_t**));
  bankwrites = (uint64_t***)calloc(n_shader, sizeof(uint64_t**));
  num_MCBs_accessed = (uint64_t*)calloc(mem_config->m_n_mem * mem_config->nbk,
                                        sizeof(uint64_t));
  if (mem_config->gpgpu_frfcfs_dram_sched_queue_size) {
    position_of_mrq_chosen = (uint64_t*)calloc(
        mem_config->gpgpu_frfcfs_dram_sched_queue_size, sizeof(uint64_t));
  } else
    position_of_mrq_chosen = (uint64_t*)calloc(1024, sizeof(uint64_t));
  for (unsigned i = 0; i < n_shader; i++) {
    bankreads[i] = (uint64_t**)calloc(mem_config->m_n_mem, sizeof(uint64_t*));
    bankwrites[i] = (uint64_t**)calloc(mem_config->m_n_mem, sizeof(uint64_t*));
    for (unsigned j = 0; j < mem_config->m_n_mem; j++) {
      bankreads[i][j] = (uint64_t*)calloc(mem_config->nbk, sizeof(uint64_t));
      bankwrites[i][j] = (uint64_t*)calloc(mem_config->nbk, sizeof(uint64_t));
    }
  }

  for (unsigned i = 0; i < mem_config->m_n_mem; i++) {
    totalbankreads[i] = (uint64_t*)calloc(mem_config->nbk, sizeof(uint64_t));
    totalbankblocked[i] = (uint64_t*)calloc(mem_config->nbk, sizeof(uint64_t));
    totalbankwrites[i] = (uint64_t*)calloc(mem_config->nbk, sizeof(uint64_t));
    totalbankaccesses[i] = (uint64_t*)calloc(mem_config->nbk, sizeof(uint64_t));
    mf_total_lat_table[i] =
        (uint64_t*)calloc(mem_config->nbk, sizeof(uint64_t));
    mf_max_lat_table[i] = (uint64_t*)calloc(mem_config->nbk, sizeof(uint64_t));
  }

  mem_access_type_stats =
      (uint64_t***)malloc(NUM_MEM_ACCESS_TYPE * sizeof(uint64_t**));
  for (unsigned i = 0; i < NUM_MEM_ACCESS_TYPE; i++) {
    mem_access_type_stats[i] =
        (uint64_t**)calloc(mem_config->m_n_mem, sizeof(uint64_t*));
    for (unsigned j = 0; (uint64_t)j < mem_config->m_n_mem; j++) {
      mem_access_type_stats[i][j] =
          (uint64_t*)calloc((mem_config->nbk + 1), sizeof(uint64_t*));
    }
  }

  L2_cbtoL2length = (uint64_t*)calloc(mem_config->m_n_mem, sizeof(uint64_t));
  L2_cbtoL2writelength =
      (uint64_t*)calloc(mem_config->m_n_mem, sizeof(uint64_t));
  L2_L2tocblength = (uint64_t*)calloc(mem_config->m_n_mem, sizeof(uint64_t));
  L2_dramtoL2length = (uint64_t*)calloc(mem_config->m_n_mem, sizeof(uint64_t));
  L2_dramtoL2writelength =
      (uint64_t*)calloc(mem_config->m_n_mem, sizeof(uint64_t));
  L2_L2todramlength = (uint64_t*)calloc(mem_config->m_n_mem, sizeof(uint64_t));
}

void memory_stats_t::init() {
  for (std::vector<App*>::iterator it = App::begin(); it != App::end(); it++) {
    (*it)->num_activates_ =
        (uint64_t**)calloc(m_memory_config->m_n_mem, sizeof(uint64_t));
    (*it)->num_activates_w_ =
        (uint64_t**)calloc(m_memory_config->m_n_mem, sizeof(uint64_t));
    (*it)->row_access_ =
        (uint64_t**)calloc(m_memory_config->m_n_mem, sizeof(uint64_t));
    (*it)->row_access_w_ =
        (uint64_t**)calloc(m_memory_config->m_n_mem, sizeof(uint64_t));
  }

  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
    for (std::vector<App*>::iterator it = App::begin(); it != App::end();
         it++) {
      (*it)->num_activates_[i] =
          (uint64_t*)calloc(m_memory_config->nbk, sizeof(uint64_t));
      (*it)->num_activates_w_[i] =
          (uint64_t*)calloc(m_memory_config->nbk, sizeof(uint64_t));
      (*it)->row_access_[i] =
          (uint64_t*)calloc(m_memory_config->nbk, sizeof(uint64_t));
      (*it)->row_access_w_[i] =
          (uint64_t*)calloc(m_memory_config->nbk, sizeof(uint64_t));
    }
  }
}

// TODO: Rachata = Mem latency stat here
// record the total latency
uint64_t memory_stats_t::memlatstat_done(mem_fetch* mf) {
  unsigned mf_latency;
  mf_latency = (gpu_sim_cycle + gpu_tot_sim_cycle) - mf->get_timestamp();
  mf_total_num_lat_pw++;
  mf_total_tot_lat_pw += mf_latency;

  if (mf->get_tlb_depth_count() > 0) {
    tlb_mf_total_num_lat_pw++;
    tlb_mf_total_tot_lat_pw += mf_latency;
  }  // TODO remove zeroth entry

  if (mf->get_sid() != (unsigned)-1) {
    App* app = App::get_app(mf->get_appID());
    app->mf_num_lat_pw++;  // the first entry is reserved for something else?
    app->mf_tot_lat_pw++;
    if (mf->get_tlb_depth_count() > 0) {
      app->tlb_mf_num_lat_pw++;
      app->tlb_mf_tot_lat_pw += mf_latency;
    }
  }

  for (std::vector<App*>::iterator it = App::begin(); it != App::end(); it++) {
    (*it)->mflatency = (float)(*it)->mf_total_lat / (*it)->num_mfs;
    (*it)->tlb_num_mfs = (float)(*it)->tlb_mf_total_lat / (*it)->tlb_num_mfs;
  }

  unsigned idx = LOGB2(mf_latency);
  assert(idx < 32);
  mf_lat_table[idx]++;
  shader_mem_lat_log(mf->get_sid(), mf_latency);
  mf_total_lat_table[mf->get_tlx_addr().chip][mf->get_tlx_addr().bk] +=
      mf_latency;

  if (mf_latency > max_mf_latency)
    max_mf_latency = mf_latency;
  if ((mf_latency > tlb_max_mf_latency) && (mf->get_tlb_depth_count() > 0))
    tlb_max_mf_latency = mf_latency;
  return mf_latency;
}

void memory_stats_t::memlatstat_read_done(mem_fetch* mf) {
  if (m_memory_config->gpgpu_memlatency_stat) {
    //      printf("Rachata-debug: in mem_latency_stat.cc, get read stat when
    //      read is done, request address = %x, level = %d\n", mf->get_addr(),
    //      mf->get_tlb_depth_count());
    unsigned mf_latency = memlatstat_done(mf);
    if (mf_latency >
        mf_max_lat_table[mf->get_tlx_addr().chip][mf->get_tlx_addr().bk])
      mf_max_lat_table[mf->get_tlx_addr().chip][mf->get_tlx_addr().bk] =
          mf_latency;
    unsigned icnt2sh_latency;
    icnt2sh_latency =
        (gpu_tot_sim_cycle + gpu_sim_cycle) - mf->get_return_timestamp();
    icnt2sh_lat_table[LOGB2(icnt2sh_latency)]++;
    if (icnt2sh_latency > max_icnt2sh_latency)
      max_icnt2sh_latency = icnt2sh_latency;
  }
}

void memory_stats_t::memlatstat_dram_access(mem_fetch* mf) {
  //   printf("Rachata-debug: in mem_latency_stat.cc, get stat after a DRAM
  //   access. Called only when a request is done, request address = %x, level =
  //   %d\n", mf->get_addr(), mf->get_tlb_depth_count());
  unsigned dram_id = mf->get_tlx_addr().chip;
  unsigned bank = mf->get_tlx_addr().bk;
  if (m_memory_config->gpgpu_memlatency_stat) {
    if (mf->get_is_write()) {
      if (mf->get_sid() < m_n_shader) {  // do not count L2_writebacks here
        bankwrites[mf->get_sid()][dram_id][bank]++;
        shader_mem_acc_log(mf->get_sid(), dram_id, bank, 'w');
      }
      totalbankwrites[dram_id][bank]++;
    } else {
      bankreads[mf->get_sid()][dram_id][bank]++;
      shader_mem_acc_log(mf->get_sid(), dram_id, bank, 'r');
      totalbankreads[dram_id][bank]++;
    }
    if (!mf->reg_dump) {
      mem_access_type_stats[mf->get_access_type()][dram_id][bank]++;
    }
  }
  if (mf->get_pc() != (unsigned)-1)
    ptx_file_line_stats_add_dram_traffic(mf->get_pc(), mf->get_data_size());

  if (mf->accum_dram_access > 0) {
    // Rachata-debug: uncomment this after done debugging mem_stat
    //       totalL1TLBMisses[mf->get_sid()][mf->get_appID()]++;
    //       if(mf->isL2TLBMiss)
    //       totalL2TLBMisses[mf->get_sid()][mf->get_appID()]++;
    //       totalTLBMissesCausedAccess[mf->get_sid()][mf->get_appID()]=totalTLBMissesCausedAccess[mf->get_sid()][mf->get_appID()]
    //       + mf->accum_dram_access;
  }
}

void memory_stats_t::memlatstat_icnt2mem_pop(mem_fetch* mf) {
  if (m_memory_config->gpgpu_memlatency_stat) {
    unsigned icnt2mem_latency;
    icnt2mem_latency =
        (gpu_tot_sim_cycle + gpu_sim_cycle) - mf->get_timestamp();
    icnt2mem_lat_table[LOGB2(icnt2mem_latency)]++;
    if (icnt2mem_latency > max_icnt2mem_latency)
      max_icnt2mem_latency = icnt2mem_latency;
  }
}

void memory_stats_t::memlatstat_lat_pw() {
  if (mf_total_num_lat_pw > 0 && m_memory_config->gpgpu_memlatency_stat) {
    mf_total_total_lat += mf_total_tot_lat_pw;
    total_num_mfs += mf_total_num_lat_pw;
    mf_lat_pw_table[LOGB2(mf_total_tot_lat_pw / mf_total_num_lat_pw)]++;
    mf_total_tot_lat_pw = 0;
    mf_total_num_lat_pw = 0;
  }
  if (m_memory_config->gpgpu_memlatency_stat) {
    for (std::vector<App*>::iterator it = App::begin(); it != App::end();
         it++) {
      (*it)->mf_total_lat += (*it)->mf_tot_lat_pw;
      (*it)->num_mfs += (*it)->mf_num_lat_pw;
      (*it)->mf_tot_lat_pw = 0;
      (*it)->mf_num_lat_pw = 0;
    }
  }

  if (tlb_mf_total_num_lat_pw && m_memory_config->gpgpu_memlatency_stat) {
    tlb_mf_total_total_lat += tlb_mf_total_tot_lat_pw;
    tlb_total_num_mfs += tlb_mf_total_num_lat_pw;
    tlb_mf_total_tot_lat_pw = 0;
    tlb_mf_total_num_lat_pw = 0;
  }
  if (m_memory_config->gpgpu_memlatency_stat) {
    for (std::vector<App*>::iterator it = App::begin(); it != App::end();
         it++) {
      (*it)->tlb_mf_total_lat += (*it)->tlb_mf_tot_lat_pw;
      (*it)->tlb_num_mfs += (*it)->tlb_mf_num_lat_pw;
      (*it)->tlb_mf_tot_lat_pw = 0;
      (*it)->tlb_mf_num_lat_pw = 0;
    }
  }
}

void memory_stats_t::memlatstat_print_file(unsigned n_mem,
                                           unsigned gpu_mem_n_bk,
                                           std::ostream& out) {
  unsigned i, j, k, l, m;
  unsigned max_bank_accesses, min_bank_accesses, max_chip_accesses,
      min_chip_accesses;

  if (m_memory_config->gpgpu_memlatency_stat) {
    out << "maxmrqlatency = " << max_mrq_latency << std::endl;
    out << "maxdqlatency = " << max_dq_latency << std::endl;
    out << "maxmflatency = " << max_mf_latency << std::endl;
    out << "high_prio_queue_drain_reset = " << drain_reset << std::endl;
    out << "average_combo_count = " << (float)total_combo / drain_reset
        << std::endl;
    out << "sched_from_normal_prio = " << sched_from_normal_prio
        << "DRAM_normal_prio = " << DRAM_normal_prio << std::endl;
    out << "sched_from_high_prio = " << sched_from_high_prio
        << "DRAM_high_prio = " << DRAM_high_prio << std::endl;
    out << "sched_from_special_prio = " << sched_from_special_prio
        << "DRAM_special_prio = " << DRAM_special_prio << std::endl;

    out << "l1_tlb_stalled_cycles_due_to_flush = {";
    for (int i = 0; i < 200; i++)
      out << TLB_L1_flush_stalled[i] << ", ";
    out << "}" << std::endl;

    out << "l1_tlb_average_number_of_warps_per_entry= {";
    for (int i = 0; i < 200; i++)
      out << TLBL1_sharer_avg[i] << ", ";
    out << "}" << std::endl;

    out << "l1_tlb_variance_number_of_warps_per_entry= {";
    for (int i = 0; i < 200; i++)
      out << TLBL1_sharer_var[i] << ", ";
    out << "}" << std::endl;

    out << "l1_tlb_max_number_of_warps_per_entry= {";
    for (int i = 0; i < 200; i++)
      out << TLBL1_sharer_max[i] << ", ";
    out << "}" << std::endl;

    out << "l1_tlb_min_number_of_warps_per_entry= {";
    for (int i = 0; i < 200; i++)
      out << TLBL1_sharer_min[i] << ", ";
    out << "}" << std::endl;

    out << "l1_tlb_number_of_unique_entries= {";
    for (int i = 0; i < 200; i++)
      out << TLBL1_total_unique_addr[i] << ", ";
    out << "}" << std::endl;

    out << "l2_tlb_average_number_of_warps_per_entry = " << TLBL2_sharer_avg
        << std::endl;
    out << "l2_tlb_variance_number_of_warps_per_entry = " << TLBL2_sharer_var
        << std::endl;
    out << "l2_tlb_max_number_of_warps_per_entry = " << TLBL2_sharer_max
        << std::endl;
    out << "l2_tlb_min_number_of_warps_per_entry = " << TLBL2_sharer_min
        << std::endl;
    out << "l2_tlb_number_of_unique_entries = " << TLBL2_total_unique_addr
        << std::endl;

    out << "l2_tlb_stalled_cycles_due_to_flush = " << TLB_L2_flush_stalled
        << std::endl;
    out << "tlb_bypass_cache_stalled_cycles_due_to_flush = "
        << TLB_bypass_cache_flush_stalled << std::endl;

    out << "l2_cache_accesses = " << l2_cache_accesses << std::endl;
    out << "l2_cache_accesses_app= {";
    for (std::vector<App*>::iterator it = App::begin(); it != App::end();
         it++) {
      out << (*it)->l2_cache_accesses_app << ", ";
    }
    out << "}" << std::endl;

    out << "l2_cache_hits = " << l2_cache_hits << std::endl;
    out << "l2_cache_hits_app= {";
    for (std::vector<App*>::iterator it = App::begin(); it != App::end();
         it++) {
      out << (*it)->l2_cache_hits_app << ", ";
    }
    out << "}" << std::endl;

    out << "l2_cache_misses = " << l2_cache_misses << std::endl;
    out << "l2_cache_misses_app= {";
    for (std::vector<App*>::iterator it = App::begin(); it != App::end();
         it++) {
      out << (*it)->l2_cache_misses_app << ", ";
    }
    out << "}" << std::endl;

    out << "number_of_coalesced_attempts = " << coalesced_tried << std::endl;
    out << "number_of_coalseced_attempts_app= {";
    for (std::vector<App*>::iterator it = App::begin(); it != App::end();
         it++) {
      out << (*it)->coalesced_tried_app << ", ";
    }
    out << "}" << std::endl;

    out << "number_of_coalesced_noinval_successes = "
        << coalesced_noinval_succeed << std::endl;
    out << "number_of_coalseced_noinval_successes_app= {";
    for (std::vector<App*>::iterator it = App::begin(); it != App::end();
         it++) {
      out << (*it)->coalesced_noinval_succeed_app << ", ";
    }
    out << "}" << std::endl;

    out << "number_of_coalesced_successes = " << coalesced_succeed << std::endl;
    out << "number_of_coalseced_successes_app= {";
    for (std::vector<App*>::iterator it = App::begin(); it != App::end();
         it++) {
      out << (*it)->coalesced_succeed_app << ", ";
    }
    out << "}" << std::endl;

    out << "number_of_coalesced_fails = " << coalesced_fail << std::endl;
    out << "number_of_coalseced_fails_app= {";
    for (std::vector<App*>::iterator it = App::begin(); it != App::end();
         it++) {
      out << (*it)->coalesced_fail_app << ", ";
    }
    out << "}" << std::endl;

    out << "number_of_coalesced_pages = " << num_coalesce << std::endl;
    out << "peak_bloated_pages = " << max_bloat << std::endl;
    out << "size_of_page_tables = " << pt_space_size << std::endl;

    out << "tlb_bypassed_data_cache = " << tlb_bypassed << std::endl;

    out << "tlb_bypassed_data_cache_app = {";
    for (std::vector<App*>::iterator it = App::begin(); it != App::end();
         it++) {
      out << (*it)->tlb_bypassed_app << ", ";
    }
    out << "}" << std::endl;

    out << "tlb_peak_occupancy_app = {";
    for (std::vector<App*>::iterator it = App::begin(); it != App::end();
         it++) {
      out << (*it)->tlb_occupancy_peak << ", ";
    }
    out << "}" << std::endl;

    out << "tlb_end_occupancy_app = {";
    for (std::vector<App*>::iterator it = App::begin(); it != App::end();
         it++) {
      out << (*it)->tlb_occupancy_end << ", ";
    }
    out << "}" << std::endl;

    out << "tlb_avg_occupancy_app = {";
    for (std::vector<App*>::iterator it = App::begin(); it != App::end();
         it++) {
      out << (*it)->tlb_occupancy_avg << ", ";
    }
    out << "}" << std::endl;

    out << "dram_req_high_queue_count = " << high_prio_queue_count << std::endl;
    for (std::vector<App*>::iterator it = App::begin(); it != App::end();
         it++) {
      out << "dram_req_high_queue_count_app" << i << " = "
          << (*it)->high_prio_queue_count_app << std::endl;
    }

    out << "dram_priority_switch_triggered = " << dram_app_switch << std::endl;
    for (std::vector<App*>::iterator it = App::begin(); it != App::end();
         it++) {
      out << "dram_priority_for_app" << i << " = "
          << (*it)->dram_prioritized_cycles_app << std::endl;
    }

    out << "tlb_level_cache_accesses = {";
    for (int i = 0; i < 10; i++)
      out << tlb_level_accesses[i] << ", ";
    out << "}" << std::endl;

    out << "tlb_level_cache_hits = {";
    for (int i = 0; i < 10; i++)
      out << tlb_level_hits[i] << ", ";
    out << "}" << std::endl;

    out << "tlb_level_cache_misses = {";
    for (int i = 0; i < 10; i++)
      out << tlb_level_misses[i] << ", ";
    out << "}" << std::endl;

    out << "tlb_level_cache_fails = {";
    for (int i = 0; i < 10; i++)
      out << tlb_level_fails[i] << ", ";
    out << "}" << std::endl;

    out << "tlb_level_cache_hit_rate = {";
    for (int i = 0; i < 10; i++)
      out << (float)tlb_level_hits[i] /
                 (float)(tlb_level_hits[i] + tlb_level_misses[i])
          << ", ";
    out << "}" << std::endl;

    if (total_num_mfs) {
      out << "averagemflatency = " << mf_total_total_lat / total_num_mfs
          << std::endl;
    }
    for (std::vector<App*>::iterator it = App::begin(); it != App::end();
         it++) {
      if ((*it)->num_mfs)  // avoid divide by zero
        out << "averagemflatency_" << i << " = "
            << (*it)->mf_total_lat / (*it)->num_mfs << std::endl;
    }

    if (tlb_total_num_mfs) {
      out << "averageTLBmflatency = "
          << tlb_mf_total_total_lat / tlb_total_num_mfs << std::endl;
    }
    for (std::vector<App*>::iterator it = App::begin(); it != App::end();
         it++) {
      if ((*it)->tlb_num_mfs)  // avoid divide by zero
        out << "averageTLBmflatency_" << i << " = "
            << (*it)->tlb_mf_total_lat / (*it)->tlb_num_mfs << std::endl;
    }

    for (std::vector<App*>::iterator it = App::begin(); it != App::end();
         it++) {
      if ((*it)->mrq_num) {
        out << "averagemrqlatency_" << i << " = "
            << (*it)->mrqs_latency / (*it)->mrq_num << std::endl;
      }
    }

    out << "max_icnt2mem_latency = " << max_icnt2mem_latency << std::endl;
    out << "max_icnt2sh_latency = " << max_icnt2sh_latency << std::endl;
    out << "mrq_lat_table:";
    for (i = 0; i < 32; i++) {
      out << mrq_lat_table[i] << ", ";
    }
    out << std::endl;
    out << "dq_lat_table:";
    for (i = 0; i < 32; i++) {
      out << dq_lat_table[i] << ", ";
    }
    out << std::endl;
    out << "mf_lat_table:";
    for (i = 0; i < 32; i++) {
      out << mf_lat_table[i] << ", ";
    }
    out << std::endl;
    out << "icnt2mem_lat_table:";
    for (i = 0; i < 24; i++) {
      out << icnt2mem_lat_table[i] << ", ";
    }
    out << std::endl;
    out << "icnt2sh_lat_table:";
    for (i = 0; i < 24; i++) {
      out << icnt2sh_lat_table[i] << ", ";
    }
    out << std::endl;
    out << "mf_lat_pw_table:";
    for (i = 0; i < 32; i++) {
      out << mf_lat_pw_table[i] << ", ";
    }
    out << std::endl;

    /*MAXIMUM CONCURRENT ACCESSES TO SAME ROW*/
    out << "maximum concurrent accesses to same row:" << std::endl;
    for (i = 0; i < n_mem; i++) {
      out << "dram[" << i << "]: ";
      for (j = 0; j < gpu_mem_n_bk; j++) {
        out << max_conc_access2samerow[i][j] << ", ";
      }
      out << std::endl;
    }

    /*MAXIMUM SERVICE TIME TO SAME ROW*/
    out << "maximum service time to same row:" << std::endl;
    for (i = 0; i < n_mem; i++) {
      out << "dram[" << i << "]: ";
      for (j = 0; j < gpu_mem_n_bk; j++) {
        out << max_servicetime2samerow[i][j] << ", ";
      }
      out << std::endl;
    }

    /*AVERAGE ROW ACCESSES PER ACTIVATE*/
    int total_row_accesses = 0;
    int total_num_activates = 0;
    std::vector<int> total_row_accesses_(ConfigOptions::n_apps, 0);
    std::vector<int> total_num_activates_(ConfigOptions::n_apps, 0);

    out << "average row accesses per activate:" << std::endl;
    for (i = 0; i < n_mem; i++) {
      out << "dram[" << i << "]: ";
      for (j = 0; j < gpu_mem_n_bk; j++) {
        total_row_accesses += row_access[i][j];
        total_num_activates += num_activates[i][j];
        int k = 0;
        for (std::vector<App*>::iterator it = App::begin(); it != App::end();
             it++) {
          total_row_accesses_[k] += (*it)->row_access_[i][j];
          total_num_activates_[k] += (*it)->num_activates_[i][j];
          k++;
        }
        out << (float)row_access[i][j] / num_activates[i][j] << ", ";
      }
      out << std::endl;
    }

    out << "average row locality = " << total_row_accesses << "/"
        << total_num_activates << " = "
        << (float)total_row_accesses / total_num_activates << std::endl;
    for (unsigned i = 0; i < ConfigOptions::n_apps; i++) {
      out << "average row locality_1 = " << total_row_accesses_[i] << "/"
          << total_num_activates_[i] << " = "
          << (float)total_row_accesses_[i] / total_num_activates_[i]
          << std::endl;
    }

    /*MEMORY ACCESSES*/
    k = 0;
    l = 0;
    m = 0;
    max_bank_accesses = 0;
    max_chip_accesses = 0;
    min_bank_accesses = 0xFFFFFFFF;
    min_chip_accesses = 0xFFFFFFFF;
    out << "number of total memory accesses made:" << std::endl;
    for (i = 0; i < n_mem; i++) {
      out << "dram[" << i << "]: ";
      for (j = 0; j < gpu_mem_n_bk; j++) {
        l = totalbankaccesses[i][j];
        if (l < min_bank_accesses)
          min_bank_accesses = l;
        if (l > max_bank_accesses)
          max_bank_accesses = l;
        k += l;
        m += l;
        out << l << ", ";
      }
      if (m < min_chip_accesses)
        min_chip_accesses = m;
      if (m > max_chip_accesses)
        max_chip_accesses = m;
      m = 0;
      out << std::endl;
    }
    out << "total accesses: " << k << std::endl;
    if (min_bank_accesses)
      out << "bank skew: " << max_bank_accesses << " /" << min_bank_accesses
          << " = " << (float)max_bank_accesses / min_bank_accesses << std::endl;
    else
      out << "min_bank_accesses = 0!" << std::endl;
    if (min_chip_accesses)
      out << "chip skew: " << max_chip_accesses << " / " << min_chip_accesses
          << " = " << (float)max_chip_accesses / min_chip_accesses << std::endl;
    else
      out << "min_chip_accesses = 0!" << std::endl;

    /*READ ACCESSES*/
    k = 0;
    l = 0;
    m = 0;
    max_bank_accesses = 0;
    max_chip_accesses = 0;
    min_bank_accesses = 0xFFFFFFFF;
    min_chip_accesses = 0xFFFFFFFF;
    out << "number of total read accesses:" << std::endl;
    for (i = 0; i < n_mem; i++) {
      out << "dram[" << i << "]: ";
      for (j = 0; j < gpu_mem_n_bk; j++) {
        l = totalbankreads[i][j];
        if (l < min_bank_accesses)
          min_bank_accesses = l;
        if (l > max_bank_accesses)
          max_bank_accesses = l;
        k += l;
        m += l;
        out << l << ", ";
      }
      if (m < min_chip_accesses)
        min_chip_accesses = m;
      if (m > max_chip_accesses)
        max_chip_accesses = m;
      m = 0;
      out << std::endl;
    }
    out << "total reads: " << k << std::endl;
    if (min_bank_accesses)
      out << "bank skew: " << max_bank_accesses << " / " << min_bank_accesses
          << " = " << (float)max_bank_accesses / min_bank_accesses << std::endl;
    else
      out << "min_bank_accesses = 0!" << std::endl;
    if (min_chip_accesses)
      out << "chip skew: " << max_chip_accesses << " / " << min_chip_accesses
          << " = " << (float)max_chip_accesses / min_chip_accesses << std::endl;
    else
      out << "min_chip_accesses = 0!" << std::endl;

    out << "number of cycles banks are stalled = {";
    for (unsigned temp1 = 0; temp1 < n_mem; temp1++) {
      for (unsigned temp2 = 0; temp2 < gpu_mem_n_bk; temp2++) {
        out << totalbankblocked[temp1][temp2] << ", ";
      }
    }
    out << "}" << std::endl;

    /*WRITE ACCESSES*/
    k = 0;
    l = 0;
    m = 0;
    max_bank_accesses = 0;
    max_chip_accesses = 0;
    min_bank_accesses = 0xFFFFFFFF;
    min_chip_accesses = 0xFFFFFFFF;
    out << "number of total write accesses:" << std::endl;
    for (i = 0; i < n_mem; i++) {
      out << "dram[" << i << "]: ";
      for (j = 0; j < gpu_mem_n_bk; j++) {
        l = totalbankwrites[i][j];
        if (l < min_bank_accesses)
          min_bank_accesses = l;
        if (l > max_bank_accesses)
          max_bank_accesses = l;
        k += l;
        m += l;
        out << l << ", ";
      }
      if (m < min_chip_accesses)
        min_chip_accesses = m;
      if (m > max_chip_accesses)
        max_chip_accesses = m;
      m = 0;
      out << std::endl;
    }
    out << "total reads: " << k << std::endl;
    if (min_bank_accesses)
      out << "bank skew: " << max_bank_accesses << " / " << min_bank_accesses
          << " = " << (float)max_bank_accesses / min_bank_accesses << std::endl;
    else
      out << "min_bank_accesses = 0!" << std::endl;
    if (min_chip_accesses)
      out << "chip skew: " << max_chip_accesses << " / " << min_chip_accesses
          << " = " << (float)max_chip_accesses / min_chip_accesses << std::endl;
    else
      out << "min_chip_accesses = 0!" << std::endl;

    /*AVERAGE MF LATENCY PER BANK*/
    out << "average mf latency per bank:" << std::endl;
    for (i = 0; i < n_mem; i++) {
      out << "dram[" << i << "]: ";
      for (j = 0; j < gpu_mem_n_bk; j++) {
        k = totalbankwrites[i][j] + totalbankreads[i][j];
        if (k)
          out << mf_total_lat_table[i][j] / k << ", ";
        else
          out << "    none  ";
      }
      out << std::endl;
    }

    /*MAXIMUM MF LATENCY PER BANK*/
    out << "maximum mf latency per bank:" << std::endl;
    for (i = 0; i < n_mem; i++) {
      out << "dram[" << i << "]: ";
      for (j = 0; j < gpu_mem_n_bk; j++) {
        out << mf_max_lat_table[i][j] << ", ";
      }
      out << std::endl;
    }
  }

  if (m_memory_config->gpgpu_memlatency_stat & GPU_MEMLATSTAT_MC) {
    out << "\nNumber of Memory Banks Accessed per Memory Operation per Warp "
           "(from 0):"
        << std::endl;
    uint64_t accum_MCBs_accessed = 0;
    uint64_t tot_mem_ops_per_warp = 0;
    for (i = 0; i < n_mem * gpu_mem_n_bk; i++) {
      accum_MCBs_accessed += i * num_MCBs_accessed[i];
      tot_mem_ops_per_warp += num_MCBs_accessed[i];
      out << num_MCBs_accessed[i] << ", ";
    }

    out << "\nAverage # of Memory Banks Accessed per Memory Operation per Warp="
        << (float)accum_MCBs_accessed / tot_mem_ops_per_warp << std::endl;

    out << "\nposition of mrq chosen" << std::endl;

    if (!m_memory_config->gpgpu_frfcfs_dram_sched_queue_size)
      j = 1024;
    else
      j = m_memory_config->gpgpu_frfcfs_dram_sched_queue_size;
    k = 0;
    l = 0;
    for (i = 0; i < j; i++) {
      out << position_of_mrq_chosen[i] << ", ";
      k += position_of_mrq_chosen[i];
      l += i * position_of_mrq_chosen[i];
    }
    out << std::endl;
    out << "\naverage position of mrq chosen = " << (float)l / k << std::endl;
  }
}

void memory_stats_t::memlatstat_print(unsigned n_mem, unsigned gpu_mem_n_bk) {
  memlatstat_print_file(n_mem, gpu_mem_n_bk, std::cout);
}
