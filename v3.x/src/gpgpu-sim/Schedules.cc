#include "Schedules.h"

#include <algorithm>
#include <cmath>
#include "ConfigOptions.h"

// Helper functions

Schedule_Assignment::Schedule_Assignment(std::map<appid_t, std::vector<uint32_t>>& sm_assignment) {
  new_assignment = sm_assignment;

  // build list of all running apps
  for (std::vector<App*>::const_iterator app = App::cbegin(); app != App::cend(); app++) {
    std::vector<uint32_t> app_sms = App::get_app_sms((*app)->appid);
    if (app_sms.size() > 0) {
      // add all sms to evicted sms list
      evicted[(*app)->appid] = app_sms;
    }
  }

  // remove to-be-scheduled app from evicted apps and new schedule, it will still be running
  for (std::map<appid_t, std::vector<uint32_t>>::iterator it = sm_assignment.begin();
      it != sm_assignment.end(); it++) {
    appid_t appid = it->first;
    if (evicted.count(appid)) {
      // remove sms in both schedules since they don't change assignment
      std::vector<uint32_t>& evicted_ref = evicted.at(appid);
      evicted.at(appid).erase(std::remove_if(evicted_ref.begin(), evicted_ref.end(),
          [it](const uint32_t& sm) {
            return std::find(it->second.begin(), it->second.end(), sm) != it->second.end();
          }), evicted_ref.end());
    }
  }

  // build list of sms that are changing to another app
  for (std::map<appid_t, std::vector<uint32_t>>::const_iterator it = sm_assignment.cbegin();
      it != sm_assignment.cend(); it++) {
    evicted_sms.insert(evicted_sms.end(), it->second.cbegin(), it->second.cend());
  }
}

/**
 * get_priority_sorted_apps sorts the entries in apps by their priority level.
 */
static std::vector<appid_t> get_priority_sorted_apps() {
  std::vector<appid_t> sorted_apps;
  for (std::vector<App*>::iterator it = App::begin(); it != App::end(); it++) {
    sorted_apps.push_back((*it)->appid);
  }
  std::sort(sorted_apps.begin(), sorted_apps.end());
  return sorted_apps;
}
/**
 * Apps each get the same number of SMs. If there are more apps than SMs, low
 * priority apps are
 * not scheduled.
 * [Future work] Impose a penalty on long-running high priority applications.
 */
Schedule_Assignment Schedules::assign_fair_schedule() {
  // Sort apps by priority
  std::vector<appid_t> sorted_apps = get_priority_sorted_apps();

  // Determine how many SMs to assign to each app
  uint32_t n_sms = ConfigOptions::n_sms;
  size_t n_apps = ConfigOptions::n_apps;
  uint32_t fair_share;
  uint32_t extra_sms;
  if (n_apps > n_sms) {
    fair_share = 1;
    extra_sms = 0;
  } else {
    fair_share = n_sms / n_apps;
    extra_sms = n_sms % n_apps;
  }

  // Create assignment of apps to SMs
  std::map<appid_t, std::vector<uint32_t>> sm_assignment;
  uint32_t current_sm = 0;
  for (std::vector<appid_t>::iterator it = sorted_apps.begin();
      it != sorted_apps.end() && current_sm < n_sms; it++) {
    appid_t appid = *it;
    std::vector<uint32_t> sms;
    if (extra_sms >
        0) {  // give each app an extra SM until there are none remaining
      sms.push_back(current_sm);
      current_sm++;
      extra_sms--;
    }
    for (uint32_t i = 0; i < fair_share;
        i++) {  // give each app their fair share of SMs
      sms.push_back(current_sm);
      current_sm++;
    }
    sm_assignment[appid] = sms;
  }
  return Schedule_Assignment(sm_assignment);
}

/**
 * We have some options on how to implement this:
 * 1. The highest priority app gets all of the SMs. If we can predict how many
 * SMs an app will
 * occupy, then an app will get min(all, occupancy) SMs and if possible the next
 * highest
 * priority app will be scheduled.
 * 2. Apps are assigned a number of SMs based on priority. An app with priority
 * i will receive
 * approximately twice as many SMs as an app with priority i + 1. Apps with the
 * same priority
 * will receive the same number of SMs.
 * There are lg(number of sms) effective priority levels, however priority is
 * presented
 * i will receive
 * approximately twice as many SMs as an app with priority i + 1. Apps with the
 * same priority
 * will receive the same number of SMs.
 * There are lg(number of sms) effective priority levels, however priority is
 * presented
 * externally as ConfigOptions::priority_max and ConfigOptions::priority_min.
 * They are internally
 * scaled to the lg(number of sms) scale.
 * If there are more apps than can be scheduled, lower priority apps will not be
 * scheduled.
 * This can lead to low priority applications not making progress.
 * [Future work] Impose a penalty on long-running high priority applications.
 */
Schedule_Assignment Schedules::assign_priority_schedule() {
  // Sort apps by priority
  std::vector<appid_t> sorted_apps = get_priority_sorted_apps();

  // Determine how many SMs to assign to each app
  uint32_t min_priority = App::get_app(sorted_apps.front())->priority;
  uint32_t n_sms = ConfigOptions::n_sms;

  std::vector<double> assignment_ratio;
  double total_assignment = 0.0;
  for (std::vector<appid_t>::iterator it = sorted_apps.begin();
      it != sorted_apps.end(); it++) {
    int priority = App::get_app(*it)->priority;
    // sm share ratio = 2^(difference of minimum priority and current priority)
    // this number will always be <= 1
    double ratio = pow(2.0, (int)min_priority - priority);
    assignment_ratio.push_back(ratio);
    total_assignment += ratio;
  }

  std::map<appid_t, std::vector<uint32_t>> sm_assignment;
  uint32_t current_sm = 0;
  std::vector<double>::iterator ratio = assignment_ratio.begin();
  for (std::vector<appid_t>::iterator it = sorted_apps.begin();
      it != sorted_apps.end() && ratio != assignment_ratio.end() &&
      current_sm < n_sms;
      it++, ratio++) {
    appid_t appid = *it;
    std::vector<uint32_t> sms;
    // n sms assigned = n_sms / (sum of assignment ratio) * (current app ratio)
    int assignment = (int)round(n_sms / total_assignment * *ratio);
    assignment = std::min(assignment, (int)(n_sms - current_sm));
    for (int i = 0; i < assignment; i++) {
      sms.push_back(current_sm);
      current_sm++;
    }
    sm_assignment[appid] = sms;
  }
  return Schedule_Assignment(sm_assignment);
}

/**
 * Schedules the app that was created first on all SMs, breaking ties by
 * priority.
 */
Schedule_Assignment Schedules::assign_fifo_schedule() {
  // Sort apps by priority
  std::vector<appid_t> sorted_apps = get_priority_sorted_apps();

  // schedule app with oldest "created_timestamp"
  appid_t oldest = *sorted_apps.begin();
  for (std::vector<appid_t>::iterator it = sorted_apps.begin();
      it != sorted_apps.end(); it++) {
    if (App::get_app(oldest)->created_timestamp >
        App::get_app(*it)->created_timestamp) {
      oldest = *it;
    }
  }

  std::vector<uint32_t> sms;
  for (uint32_t i = 0; i < ConfigOptions::n_sms; i++) {
    sms.push_back(i);
  }
  std::map<appid_t, std::vector<uint32_t>> sm_assignment;
  sm_assignment[oldest] = sms;
  return Schedule_Assignment(sm_assignment);
}

/**
 * Schedules the app that comes after the currently executing app in a priority
 * sort.
 */
Schedule_Assignment Schedules::assign_rr_schedule() {
  // Sort apps by priority
  std::vector<appid_t> sorted_apps = get_priority_sorted_apps();

  // See which app is currently running
  std::vector<appid_t>::iterator next = sorted_apps.end();
  for (std::vector<appid_t>::iterator it = sorted_apps.begin();
      it != sorted_apps.end(); it++) {
    if (App::get_app_sms(*it).size()) {
      // this app is currently running
      next = it + 1;
      break;
    }
  }
  if (next == sorted_apps.end()) {
    // run the first app
    next = sorted_apps.begin();
  }
  std::vector<uint32_t> sms;
  for (uint32_t i = 0; i < ConfigOptions::n_sms; i++) {
    sms.push_back(i);
  }
  std::map<appid_t, std::vector<uint32_t>> sm_assignment;
  sm_assignment[*next] = sms;
  return Schedule_Assignment(sm_assignment);
}

/**
 * Shortest job first schedule.
 */
Schedule_Assignment Schedules::assign_sjf_schedule() {
  // TODO

  // This will require a cost predictor for the length of a job
  // Do we have anything like that? Do we get this information from profiling
  // repeated runs
  // of an app?
  assert(false);
}

Schedule_Assignment Schedules::assign_lifo_schedule() {
  assert(false);
}
