// Copyright (c) 2009-2011, Tor M. Aamodt
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

#include "mem_fetch.h"
#include "gpu-sim.h"
#include "mem_latency_stat.h"
#include "shader.h"
#include "visualizer.h"
class memory_config;

unsigned mem_fetch::sm_next_mf_request_uid = 1;

mem_fetch::mem_fetch(mem_fetch* mf) : m_appID(mf->get_appID()) {
  pwcache_hit = false;
  pwcache_done = false;

  m_request_uid = sm_next_mf_request_uid++;

  // backward reference to the original memory request -- Fixme
  m_access = mf->get_access();

  m_tlb_depth_count = mf->get_tlb_depth_count() + 1;

  m_tlb = mf->get_tlb();
  new_addr_type new_addr =
      ((uint64_t)App::get_app(mf->get_appID())->addr_offset) << 48 |
      m_tlb->get_tlbreq_addr(mf);

  set_addr(new_addr);

  m_inst = mf->get_inst();
  assert(mf->get_wid() == m_inst.warp_id());

  m_data_size = mf->get_data_size();
  m_ctrl_size = mf->get_ctrl_size();
  m_sid = mf->get_sid();
  m_tpc = mf->get_tpc();
  m_wid = mf->get_wid();

  // The address mapping starts here
  // Assumption --> mf contains appID and actual address already

  m_tlb_related_req = true;

  m_tlb_miss = false;

  m_original_addr = mf->get_original_addr();

  m_mem_config = mf->get_mem_config();

  set_page_fault(m_mem_config->m_address_mapping.addrdec_tlx(
      m_access.get_addr(), &m_raw_addr, m_appID, m_tlb_depth_count,
      !m_access.is_write()));

  m_partition_addr = m_mem_config->m_address_mapping.partition_address(
      m_access.get_addr(), m_appID, m_tlb_depth_count, !m_access.is_write());

  m_type = READ_REQUEST;
  m_timestamp = gpu_sim_cycle + gpu_tot_sim_cycle;
  m_timestamp2 = 0;
  m_status = MEM_FETCH_INITIALIZED;
  m_status_change = gpu_sim_cycle + gpu_tot_sim_cycle;
  icnt_flit_size = m_mem_config->icnt_flit_size;

  beenThroughL1 = false;

  if (mf->get_cache() == NULL) {
    assert(false);
  } else
    m_cache = mf->get_cache();

  accum_dram_access = 0;

  m_parent_tlb_request = mf;

  bypass_L2 = false;
  m_page_fault = false;

  // TODO: Rachata: Check this: Set an appropriate flags for this mem_fetch.
  // Probe the cache in case this memory access is already in the cache

  isL2TLBMiss = false;
}

mem_fetch::mem_fetch(const mem_access_t& access,
                     const warp_inst_t* inst,
                     unsigned ctrl_size,
                     unsigned wid,
                     unsigned sid,
                     unsigned tpc,
                     const memory_config* config)
    : m_appID(App::get_app_id_from_sm(sid)) {
  m_request_uid = sm_next_mf_request_uid++;
  m_access = access;
  if (inst) {
    m_inst = *inst;
    assert(wid == m_inst.warp_id());
  }
  m_data_size = access.get_size();
  m_ctrl_size = ctrl_size;
  m_sid = sid;
  m_tpc = tpc;
  m_wid = wid;
  m_tlb = NULL;  // By default, this should be null
  // TODO: Rachata -- Somehow calling find_app(sid) here cause segfault
  // config->m_address_mapping.addrdec_tlx(access.get_addr(),&m_raw_addr, 0);
  // m_partition_addr =
  // config->m_address_mapping.partition_address(access.get_addr(), 0);
  m_type = m_access.is_write() ? WRITE_REQUEST : READ_REQUEST;
  m_timestamp = gpu_sim_cycle + gpu_tot_sim_cycle;
  m_timestamp2 = 0;
  m_status = MEM_FETCH_INITIALIZED;
  m_status_change = gpu_sim_cycle + gpu_tot_sim_cycle;
  m_mem_config = config;
  icnt_flit_size = config->icnt_flit_size;

  m_original_addr = m_access.get_addr();

  m_tlb_related_req = false;

  m_cache = NULL;

  set_page_fault(config->m_address_mapping.addrdec_tlx(
      access.get_addr(), &m_raw_addr, m_appID, 0, !m_access.is_write()));
  m_partition_addr = config->m_address_mapping.partition_address(
      access.get_addr(), m_appID, 0, !m_access.is_write());

  m_tlb_depth_count = 0;
  // set_tlb_miss(false);
  m_tlb_miss = false;

  m_page_fault = false;

  m_parent_tlb_request = NULL;

  accum_dram_access = 0;

  // Rachata: Only set from the L2 TLB, if this is a TLB miss at L2 TLB
  isL2TLBMiss = false;

  bypass_L2 = false;
  pwcache_hit = false;
  pwcache_done = false;
  beenThroughL1 = false;
}

/* This constructor is used to construct a memory request for dumping/loading the register state */
mem_fetch::mem_fetch(const memory_config* config, new_addr_type address, unsigned size, unsigned sid,
    mf_type type) : reg_dump(true), m_appID(App::get_app_id_from_sm(sid)), m_type(type),
    m_mem_config(config) {
  m_original_addr = address;
  m_sid = sid;
  m_timestamp = gpu_sim_cycle + gpu_tot_sim_cycle;
  m_timestamp2 = 0;
  m_status = IN_PARTITION_L2_TO_DRAM_QUEUE;

  set_page_fault(m_mem_config->m_address_mapping.addrdec_tlx(m_original_addr, &m_raw_addr, m_appID,
        0, m_type == READ_REQUEST));

  m_tlb_depth_count = 0;
  m_partition_addr = m_raw_addr.chip;

  m_tlb_depth_count = 0;
  m_tlb_miss = false;
  m_page_fault = false;
  m_parent_tlb_request = NULL;
  accum_dram_access = 0;
  isL2TLBMiss = false;
  bypass_L2 = false;
  pwcache_hit = false;
  pwcache_done = false;
  beenThroughL1 = false;
}

mem_fetch::~mem_fetch() {
  m_status = MEM_FETCH_DELETED;
}

#define MF_TUP_BEGIN(X) static const char* Status_str[] = {
#define MF_TUP(X) #X
#define MF_TUP_END(X) \
  }                   \
  ;
#include "mem_fetch_status.tup"
#undef MF_TUP_BEGIN
#undef MF_TUP
#undef MF_TUP_END

void mem_fetch::set_page_fault(bool val) {
  m_page_fault = val;
}

// Check if the request is a write, This is rewritten to make sure any
// TLB-related req returns as a read.
bool mem_fetch::is_write() {
  if (m_tlb_depth_count > 0)
    return false;
  else
    return m_access.is_write();
}

// Check if the request is a write, This is rewritten to make sure any
// TLB-related req returns as a read. For some stupid reasons gpgpu_sim has both
// is_write and get_is_write ...
bool mem_fetch::get_is_write() const {
  if (reg_dump) {
    return m_type != READ_REQUEST;
  }
  if (m_tlb_depth_count > 0)
    return false;
  else
    return m_access.is_write();
}

void mem_fetch::set_cache(data_cache* cache) {
  m_cache = cache;
}
data_cache* mem_fetch::get_cache() {
  return m_cache;
}

unsigned mem_fetch::get_subarray() {
  return m_raw_addr.subarray;
}

bool mem_fetch::check_bypass(mem_fetch* mf) {
  //    if(m_mem_config->tlb_bypass_enabled == 1 && ( mf->get_tlb_depth_count()
  //    <= m_mem_config->tlb_bypass_level)) //Static policy
  if (m_mem_config->tlb_bypass_enabled &&
      (mf->get_tlb_depth_count() >=
       m_mem_config->tlb_bypass_level))  // Old one with inverse PT walking
  {
    return true;
  } else if (m_mem_config->tlb_bypass_enabled == 2) {
    unsigned curr_level = mf->get_tlb_depth_count();
    if (mf->get_tlb()
            ->get_shared_tlb()
            ->get_mem_stat()
            ->tlb_level_hits[curr_level] < 100)
      return false;  // Disable bypassing if there is not enough request to keep
                     // track, or still warming up

    float hit_rate = (float)(mf->get_tlb()
                                 ->get_shared_tlb()
                                 ->get_mem_stat()
                                 ->tlb_level_hits[curr_level]) /
                     (float)(mf->get_tlb()
                                 ->get_shared_tlb()
                                 ->get_mem_stat()
                                 ->tlb_level_accesses[curr_level]);
    if (hit_rate < (float)m_mem_config->data_cache_bypass_threshold / 100.0)
      return true;
    else
      return false;
  } else if (m_mem_config->tlb_bypass_enabled == 3) {
    // TODO: Add dynamic policy, check level hit rate vs. shared cache hit rate
    if ((float)(mf->get_tlb()
                    ->get_shared_tlb()
                    ->get_mem_stat()
                    ->tlb_level_hits[mf->get_tlb_depth_count()]) /
            (float)(mf->get_tlb()
                        ->get_shared_tlb()
                        ->get_mem_stat()
                        ->tlb_level_accesses
                            [mf->get_tlb_depth_count()]) /*TLB level hit rate*/
        <
        (float)(mf->get_tlb()
                    ->get_shared_tlb()
                    ->get_mem_stat()
                    ->l2_cache_hits) /
            (float)(mf->get_tlb()
                        ->get_shared_tlb()
                        ->get_mem_stat()
                        ->l2_cache_accesses) /*L2 data cache hit rate*/)  // Can
                                                                          // get
      // from
      // l2cache.cc
      // line
      // 770,
      // or
      // just
      // collect
      // it
      // again
      return true;
    else
      return false;
  } else {
    return false;
  }
}

// Rachata: Note: when the request is serviced in DRAM, invoke this routine
// again on the parent
//               This code is on l2cache.cc in dram_cycle();
// This routine is invoked only with a tlb request got its data from DRAM (at
// the DRAM return queue pop)
void mem_fetch::done_tlb_req(mem_fetch* mf) {
  // TODO: Rachata -- Add a step here to skip level?

  mem_fetch* mfr;
  mfr = mf;
  if (mfr->get_parent_tlb_request() != NULL) {
    // If PW cache hit
    if (mfr->pwcache_hit && !mfr->pwcache_done) {
      // Put this request in the latency queue for a PW cache hit
      mfr->get_tlb()->get_shared_tlb()->pw_cache_lat_queue->push_front(mfr);
      mfr->get_tlb()->get_shared_tlb()->pw_cache_lat_time->push_front(
          mfr->get_timestamp());
      return;
    } else if (mfr->pwcache_hit && mfr->pwcache_done) {
      done_tlb_req(mfr->get_parent_tlb_request());
      delete mfr;  // Not being used after this
      return;
    }

    mfr->get_parent_tlb_request()->accum_dram_access =
        mfr->accum_dram_access + 1;  // stats collection

    mfr->bypass_L2 = mfr->check_bypass(mfr);

    m_cache->add_tlb_miss_to_queue(
        mfr);  // Send the actual (original) memory request to DRAM
  } else {
    mf->been_through_tlb = true;

    mf->set_tlb_miss(false);

    mf->get_tlb()->fill(mf->get_addr(), mf);
    mf->get_tlb()->l2_fill(mf->get_addr(), mf->get_appID(), mf);

    // Rachata --- Do not have to send this actual request anymore because
    // pt_walk would return reservation fail --> The mf is deleted.
    mf->get_tlb()->add_mf_to_done(mf);
  }
}

void mem_fetch::probe_pw_cache(mem_fetch* mf) {
  if (mf->get_tlb()->get_shared_tlb()->pw_cache_access(mf)) {
    mf->pwcache_hit = true;
  }
}

// Setup dependent memory requests for DRAM to be serviced for a TLB miss
void mem_fetch::pt_walk(mem_fetch* mf) {
  if (mf->get_mem_config()->tlb_fixed_latency_enabled) {
    mf->get_tlb()->get_shared_tlb()->put_mf_to_static_queue(mf);
    return;
  }

  // Rachata-debug: Check if the creation of memory requests from TLB miss is
  // working or not
  //        m_child = new mem_fetch(mf); // Rachata-debug: Force to create a
  //        dummy memory request
  ////        mf->get_DRAM()->push(mf); //Rachata-debug: Force to push another
  /// dummy requests to DRAM
  //        m_child->get_DRAM()->push(m_child); //Rachata-debug: Force to push
  //        this dummy request to DRAM
  //        mf->done_tlb_req(mf); // Rachata-debug: Done with TLB request right
  //        away, need to comment/delete this once done checking pt_walk routine

  if (mf->get_tlb_depth_count() >=
      m_mem_config->tlb_levels)  // Done with setting up the last request in the
                                 // PT walk routine
  {
    mf->done_tlb_req(
        mf);  // Start to push requests for TLBs to DRAM level-by-level
  } else {
    // TODO: Rachata -- Add a step here to skip level?

    mem_fetch* child;
    // Assume this constructor will probe the TLB and set the TLB status
    // properly
    // TODO: Rachata: Need to check this part (on TLB hit --> what to do, on
    // miss --> what else to do), this is in the constructor of mem_fetch.cc
    //        if(mf->get_parent_tlb_request()!=NULL) printf("Rachata-debug: in
    //        mem_fetch.cc, parent request is to address = %x\n",
    //        mf->get_parent_tlb_request()->get_addr());
    child =
        new mem_fetch(mf);  // set a new mem_fetch for the next level subroutine

    // Page walk cache here, probe the PW cache in case it is a hit, on hit,
    // mark the mf that it is a PW cache hit
    if (m_mem_config->pw_cache_enable)
      probe_pw_cache(child);

    set_child_tlb_request(child);
    pt_walk(get_child_tlb_request());  // Then, continue performing the page
                                       // table walk for the next level of TLB
                                       // access
  }
}

std::ostream& operator<<(std::ostream& o, const mem_fetch& m) {
  o << "mf: uid=" << m.m_request_uid << ", sid" << m.m_sid << ":w" << m.m_wid
      << ", part=" << m.m_raw_addr.chip << ", ";
  o << m.m_access;
  if ((unsigned)m.m_status < NUM_MEM_REQ_STAT)
    o << " status = " << Status_str[m.m_status] << " (" << m.m_status_change
        << ") ";
  else
    o << " status = " << m.m_status << " (" << m.m_status_change << "), ";
  if (!m.m_inst.empty())
    o << m.m_inst;
  o << std::endl;
  return o;
}

void mem_fetch::set_status(enum mem_fetch_status status,
                           unsigned long long cycle) {
  m_status = status;
  m_status_change = cycle;
}

bool mem_fetch::isatomic() const {
  if (m_inst.empty())
    return false;
  return m_inst.isatomic();
}

void mem_fetch::do_atomic() {
  m_inst.do_atomic(m_access.get_warp_mask());
}

bool mem_fetch::istexture() const {
  if (m_inst.empty())
    return false;
  return m_inst.space.get_type() == tex_space;
}

bool mem_fetch::isconst() const {
  if (m_inst.empty())
    return false;
  return (m_inst.space.get_type() == const_space) ||
         (m_inst.space.get_type() == param_space_kernel);
}

/// Returns number of flits traversing interconnect. simt_to_mem specifies the
/// direction
unsigned mem_fetch::get_num_flits(bool simt_to_mem) {
  unsigned sz = 0;
  // If atomic, write going to memory, or read coming back from memory, size =
  // ctrl + data. Else, only ctrl
  if (isatomic() || (simt_to_mem && get_is_write()) ||
      !(simt_to_mem || get_is_write()))
    sz = size();
  else
    sz = get_ctrl_size();

  return (sz / icnt_flit_size) + ((sz % icnt_flit_size) ? 1 : 0);
}
