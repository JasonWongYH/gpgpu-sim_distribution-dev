/*
 * App.h
 *
 *  Created on: Mar 3, 2017
 *      Author: vance
 */

#ifndef SRC_GPGPU_SIM_APP_H_
#define SRC_GPGPU_SIM_APP_H_

#include <stdint.h>
#include <functional>
#include <list>
#include <map>
#include <set>
#include <vector>
#include "../abstract_hardware_model.h"  // For new_addr_type...

// Forward declarations
class opndcoll_rfu_t;

class appid_t {
 public:
  appid_t() : valid(false) {}
  static appid_t create() { return appid_t(next_identifier++); }
  bool operator==(const appid_t& other) const {
    assert(valid);
    return this->my_id == other.my_id;
  }
  bool operator!=(const appid_t& other) const {
    assert(valid);
    return this->my_id != other.my_id;
  }
  bool operator<(const appid_t& other) const {
    assert(valid);
    return this->my_id < other.my_id;
  }
  bool operator>(const appid_t& other) const {
    assert(valid);
    return this->my_id > other.my_id;
  }
  friend std::ostream& operator<<(std::ostream& os, const appid_t& appid);
  friend struct std::hash<appid_t>;

 protected:
 private:
  appid_t(uint32_t my_id) : my_id(my_id), valid(true) {};
  static uint32_t next_identifier;
  uint32_t my_id;
  bool valid;
};

namespace std {
// hash function for appid_t
template <>
struct hash<appid_t> {
  size_t operator()(const appid_t& o) const {
    return std::hash<int>()(o.my_id);
  }
};
}

class App {
 public:
  static App* get_app(appid_t);
  static appid_t get_app_id_from_sm(uint32_t);
  static appid_t get_app_id_from_thread(void* tid);
  static appid_t create_app(int priority);
  static appid_t register_app(int, int);
  static size_t get_created_app_count(void);
  static const std::vector<uint32_t> get_app_sms(appid_t);
  static void set_app_sms(std::map<appid_t, std::vector<uint32_t>>& app_sms);

  // Iterator functions
  static std::vector<App*>::iterator begin(void) { return app_vector.begin(); }
  static std::vector<App*>::iterator end(void) { return app_vector.end(); }
  static std::vector<App*>::const_iterator cbegin(void) {
    return app_vector.cbegin();
  }
  static std::vector<App*>::const_iterator cend(void) {
    return app_vector.cend();
  }

  // Special apps for memory operations requiring an Appid
  static App noapp;
  static App pt_space;
  static App mixapp;
  static App prefrag;

 private:
  static std::map<appid_t, App*> app_map;
  static std::vector<App*> app_vector;
  static std::map<uint32_t, appid_t> sm_to_app;
  static std::map<void*, appid_t> thread_id_to_app_id;

 private:
  App(appid_t appid, uint8_t priority, FILE* output);

 public:
  virtual ~App();

  bool operator<(const App& other) const {
    return this->priority < other.priority;
  }

  friend std::ostream& operator<<(std::ostream& o, const App& a) {
    return o << "App " << a.appid << " created " << a.created_timestamp;
  }

 public:
  const appid_t appid;
  // for scheduler
  const uint8_t priority;
  const time_t created_timestamp;
  std::map<uint32_t, opndcoll_rfu_t*> sm_to_operand_collector;

  uint64_t gpu_sim_instruction_count;
  uint64_t gpu_total_simulator_cycles_stream;

  float periodic_l2mpki;
  float periodic_miss_rate;
  float mflatency;
  float tlb_mflatency;

  // From stream_manager
  bool stat_flag;
  uint64_t app_insn;
  FILE* output;

  // From dram
  uint64_t n_req;
  uint64_t bwutil;
  uint64_t bwutil_data;
  uint64_t bwutil_tlb;
  uint64_t bwutil_periodic;
  uint64_t bwutil_periodic_data;
  uint64_t bwutil_periodic_tlb;
  uint64_t n_cmd_blp;
  uint64_t mem_state_blp;
  uint64_t dram_cycles_active;
  uint64_t blp;
  uint64_t k_app;
  float miss_rate_d;

  // From dram_sched
  uint64_t epoch_app_concurrent;

  // From tlb
  float tokens;  // tlb bypass tokens
  float total_tokens;
  bool wid_tokens[4000];
  unsigned epoch_accesses;
  unsigned epoch_hit;
  unsigned epoch_bypass_hit;
  unsigned epoch_miss;
  float epoch_previous_miss_rate;
  float epoch_previous2_miss_rate;
  unsigned flush_count;
  unsigned total_access_cache;  // To count how many cache accesses are there.
                                // For flushing
  uint64_t concurrent_tracker;
  unsigned wid_epoch_accesses[4000];
  unsigned wid_epoch_hit[4000];
  unsigned wid_epoch_miss[4000];
  float wid_epoch_previous_miss_rate[4000];
  float wid_epoch_previous2_miss_rate[4000];
  std::list<new_addr_type>* miss_tracker;
  std::map<new_addr_type, unsigned>* miss_tracker_count;
  std::list<unsigned long long>* miss_tracker_timestamp;
  unsigned long long tlb_occupancy;

  // From tlb_tag_array
  std::set<new_addr_type> addr_mapping;
  bool evicted;

  // From memory_stats_t

  uint64_t mrqs_latency;
  uint64_t mrq_num;
  uint64_t mf_num_lat_pw;
  uint64_t tlb_mf_num_lat_pw;
  uint64_t mf_tot_lat_pw;  // total latency summed up per window.
  // divide by mf_num_lat_pw to obtain average latency Per Window
  uint64_t tlb_mf_tot_lat_pw;  // total latency summed up per window.
  // divide by mf_num_lat_pw to obtain average latency Per Window
  uint64_t mf_total_lat;
  uint64_t tlb_mf_total_lat;
  uint64_t high_prio_queue_count_app;
  uint64_t coalesced_tried_app;
  uint64_t coalesced_succeed_app;
  uint64_t coalesced_noinval_succeed_app;
  uint64_t coalesced_fail_app;
  uint64_t tlb_bypassed_app;
  uint64_t l2_cache_accesses_app;
  uint64_t l2_cache_hits_app;
  uint64_t l2_cache_misses_app;
  float tlb_occupancy_end;
  float tlb_occupancy_peak;
  float tlb_occupancy_avg;
  uint64_t dram_prioritized_cycles_app;  // How many cycles a certain app is
                                         // prioritized in DRAM
  uint64_t num_mfs;
  uint64_t tlb_num_mfs;
  float rbl;
  uint64_t lat;
  uint64_t** num_activates_;
  uint64_t** row_access_;
  uint64_t** num_activates_w_;
  uint64_t** row_access_w_;

  // From shader
  uint64_t pw_cache_hit_app;
  uint64_t pw_cache_miss_app;
  uint64_t tlb_hit_app;
  uint64_t large_tlb_hit_app;
  uint64_t small_tlb_hit_app;
  uint64_t tlb2_hit_app;
  uint64_t large_tlb2_hit_app;
  uint64_t small_tlb2_hit_app;
  uint64_t tlb_fault_app;
  uint64_t tlb2_fault_app;
  uint64_t tlb_miss_app;
  uint64_t large_tlb_miss_app;
  uint64_t small_tlb_miss_app;
  uint64_t tlb2_miss_app;
  uint64_t large_tlb2_miss_app;
  uint64_t small_tlb2_miss_app;
  uint64_t tlb_access_app;
  uint64_t tlb2_access_app;
  uint64_t tlb_prefetch_hit_app;
  // uint64_t tlb_bypassed_app;
  uint64_t large_tlb_bypassed_app;
  uint64_t small_tlb_bypassed_app;
  uint64_t tlb_concurrent_serviced_app;
  uint64_t tlb_current_concurrent_serviced_app;
  uint64_t avail_warp_app;
  uint64_t tlb_hit_app_epoch;
  uint64_t tlb2_hit_app_epoch;
  uint64_t tlb_miss_app_epoch;
  uint64_t tlb2_miss_app_epoch;
  uint64_t tlb_access_app_epoch;
  uint64_t tlb2_access_app_epoch;
  uint64_t tlb_concurrent_max_app;
  uint64_t l1cache_hit_app_epoch;
  uint64_t l2cache_hit_app_epoch;
  uint64_t l1cache_miss_app_epoch;
  uint64_t l2cache_miss_app_epoch;
  uint64_t l1cache_access_app_epoch;
  uint64_t l2cache_access_app_epoch;
  float available_warp_per_tlb_app;

  // from gpu-cache
  uint64_t m_access_s;
  uint64_t m_miss_s;
  uint64_t m_access_s_previous;
  uint64_t m_miss_s_previous;

  // From mem_fetch
  uint32_t addr_offset;
};

#endif /* SRC_GPGPU_SIM_APP_H_ */
