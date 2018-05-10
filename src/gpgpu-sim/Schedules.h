/*
 * Schedules.h
 *
 *  Created on: June 24, 2017
 *      Author: vance
 */

#ifndef SRC_GPGPU_SIM_SCHEDULES_H_
#define SRC_GPGPU_SIM_SCHEDULES_H_

#include "App.h"

#include <stdint.h>
#include <map>
#include <utility>
#include <vector>

struct Schedule_Assignment {
  Schedule_Assignment() {};
  Schedule_Assignment(std::map<appid_t, std::vector<uint32_t>>& sm_assignment);
  std::map<appid_t, std::vector<uint32_t>> new_assignment;
  std::map<appid_t, std::vector<uint32_t>> evicted;
  std::vector<uint32_t> evicted_sms;
};

class Schedules {
 public:
  enum context_switch_state {
    NORMAL, SM_DISABLE, PIPE_DRAIN, REG_DUMP, REG_DUMP_DRAIN, CACHE_DUMP, CACHE_DUMP_DRAIN,
    REG_LOAD, SM_ENABLE
  };

  static Schedule_Assignment assign_fair_schedule(void);
  static Schedule_Assignment assign_priority_schedule(void);
  static Schedule_Assignment assign_fifo_schedule(void);
  static Schedule_Assignment assign_rr_schedule(void);
  static Schedule_Assignment assign_sjf_schedule(void);
  static Schedule_Assignment assign_lifo_schedule(void);
};

#endif /* SRC_GPGPU_SIM_SCHEDULES_H_ */
