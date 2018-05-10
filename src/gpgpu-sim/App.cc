/*
 * App.cc
 *
 *  Created on: Mar 3, 2017
 *      Author: vance
 */

#include "App.h"
#include <pthread.h>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include "ConfigOptions.h"

App::App(appid_t appid, uint8_t priority, FILE* output)
    : appid(appid),
      priority(priority),
      created_timestamp(time(0)),
      output(output) {}

App::~App() {
  if (output)
    fclose(output);
}

// Definition of static members
uint32_t appid_t::next_identifier = 666;  // arbitrary
std::map<appid_t, App*> App::app_map;
std::vector<App*> App::app_vector;
std::map<uint32_t, appid_t> App::sm_to_app;
std::map<void*, appid_t> App::thread_id_to_app_id;
// special apps
App App::noapp(appid_t::create(), ConfigOptions::default_priority, NULL);
App App::pt_space(appid_t::create(), ConfigOptions::default_priority, NULL);
App App::mixapp(appid_t::create(), ConfigOptions::default_priority, NULL);
App App::prefrag(appid_t::create(), ConfigOptions::default_priority, NULL);

std::ostream& operator<<(std::ostream& os, const appid_t& appid) {
  os << appid.my_id;
  return os;
}

const std::vector<uint32_t> App::get_app_sms(appid_t appid) {
  std::vector<uint32_t> sms;
  for (std::map<uint32_t, appid_t>::const_iterator i = App::sm_to_app.cbegin();
       i != App::sm_to_app.cend(); i++) {
    if (i->second == appid) {
      sms.push_back(i->first);
    }
  }
  return sms;
}

/**
 * Assigns each sm in sms to appid.
 *
 * This function reassigns all appids to new sms.
 */
void App::set_app_sms(std::map<appid_t, std::vector<uint32_t>>& app_sms) {
  App::sm_to_app.clear();
  for (std::map<appid_t, std::vector<uint32_t>>::const_iterator it = app_sms.cbegin();
      it != app_sms.cend(); it++) {
    for (std::vector<uint32_t>::const_iterator sm = (*it).second.cbegin();
        sm != (*it).second.cend(); sm++) {
      App::sm_to_app[*sm] = (*it).first;
    }
  }
}

appid_t App::get_app_id_from_sm(uint32_t sm_number) {
  return sm_to_app.at(sm_number);
}

appid_t App::get_app_id_from_thread(void* tid) {
  return thread_id_to_app_id.at(tid);
}

App* App::get_app(appid_t app) {
  return App::app_map.at(app);
}

size_t App::get_created_app_count(void) {
  return App::app_map.size();
}

appid_t App::create_app(int priority) {
  static uint32_t addr_offset = 0;
  appid_t my_id = appid_t::create();
  thread_id_to_app_id.insert(
      std::pair<void*, appid_t>((void*)pthread_self(), my_id));
  std::stringstream fname;
  fname << "app-" << my_id << "-out.txt";
  FILE* output = fopen(fname.str().c_str(), "w");
  App::app_map[my_id] = new App(my_id, priority, output);
  App::app_map[my_id]->addr_offset = addr_offset++;
  App::app_vector.push_back(App::app_map[my_id]);
  return my_id;
}
