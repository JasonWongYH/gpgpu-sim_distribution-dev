/*
 * ConfigOptions.cc
 *
 *  Created on: Feb 16, 2017
 *      Author: vance
 */

#include "ConfigOptions.h"

namespace ConfigOptions {
uint32_t n_apps = 0;  // To be initialized elsewhere
uint32_t n_sms = 32;  // as was previously defined in gpu-sim.cc
uint32_t schedule_cycles =
    10000;  // Run the scheduler every hundred thousand cycles
uint32_t default_priority = 10;  // between max and min
uint32_t priority_max = 1;       // low value
uint32_t priority_min = 20;      // high value
};
