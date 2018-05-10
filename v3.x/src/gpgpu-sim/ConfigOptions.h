/*
 * ConfigOptions.h
 *
 *  Created on: Feb 16, 2017
 *      Author: vance
 */

#ifndef SRC_GPGPU_SIM_CONFIGOPTIONS_H_
#define SRC_GPGPU_SIM_CONFIGOPTIONS_H_

#include <stdint.h>

namespace ConfigOptions {
extern uint32_t n_apps;
extern uint32_t n_sms;
extern uint32_t schedule_cycles;
extern uint32_t default_priority;  // between max and min
extern uint32_t priority_max;      // low value
extern uint32_t priority_min;      // high value
};

#endif /* SRC_GPGPU_SIM_CONFIGOPTIONS_H_ */
