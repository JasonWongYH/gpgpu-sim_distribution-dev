// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung
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

#include "stream_manager.h"
#include <sstream>
#include <string>
#include <vector>
#include "cuda-sim/cuda-sim.h"
#include "gpgpu-sim/ConfigOptions.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpusim_entrypoint.h"

#define DEBUG_ENABLE 0
#define STREAM_DEBUG if (DEBUG_ENABLE) \
  std::cout << "STREAM: "

unsigned CUstream_st::next_stream_id = 0;

CUstream_st::CUstream_st() {
  this->current_op_pending = false;
  this->one_time_only = false;
  this->stream_id = CUstream_st::next_stream_id++;
  pthread_mutex_init(&this->stream_lock, NULL);
}

unsigned CUstream_st::get_stream_id() { return this->stream_id; }

bool CUstream_st::empty() { return this->operations.empty(); }

bool CUstream_st::busy() { return this->current_op_pending; }

void CUstream_st::synchronize() {
  while (!this->empty())
    ;  // spin wait for operations to empty
}

void CUstream_st::push(const stream_operation &op) {
  if (op.is_done_once()) {
    this->one_time_only = 1;
  }
  pthread_mutex_lock(&this->stream_lock);
  this->operations.push_back(op);
  pthread_mutex_unlock(&this->stream_lock);
}

void CUstream_st::record_next_done() {
  // called by gpu thread
  pthread_mutex_lock(&this->stream_lock);
  assert(this->current_op_pending);
  this->operations.pop_front();
  this->current_op_pending = false;
  pthread_mutex_unlock(&this->stream_lock);
}

stream_operation CUstream_st::next() {
  // called by gpu thread
  this->current_op_pending = true;
  return this->operations.front();
}

std::ostream& operator<<(std::ostream& o, const CUstream_st& s) {
  pthread_mutex_lock(&(s.stream_lock));
  o << "GPGPU-Sim API:    stream " << s.stream_id << " has "
      << s.operations.size() << " operations" << std::endl;
  unsigned n = 0;
  for (std::list<stream_operation>::const_iterator i = s.operations.cbegin();
       i != s.operations.cend(); i++) {
    o << "GPGPU-Sim API:       " << n++ << " : " << *i << std::endl;
  }
  pthread_mutex_unlock(&(s.stream_lock));
  return o;
}

void stream_operation::do_operation(gpgpu_sim *gpu) {
  if (is_noop()) return;

  assert(!m_done && m_stream);
  STREAM_DEBUG << "GPGPU-Sim API: stream " << m_stream->get_stream_id()
              << " performing ";
  switch (m_type) {
    case stream_memcpy_host_to_device:
      STREAM_DEBUG << "memcpy host-to-device (device addr 0x" << std::hex <<
          m_device_address_dst << ", host addr 0x" << m_host_address_src <<
          std::dec << ", count = " << m_cnt << ")" << std::endl;
      gpu->memcpy_to_gpu(m_device_address_dst, m_host_address_src, m_cnt);
      m_stream->record_next_done();
      break;
    case stream_memcpy_device_to_host:
      STREAM_DEBUG << "memcpy device-to-host (device addr 0x" << std::hex <<
          m_device_address_src << ", host addr 0x" << m_host_address_dst <<
          std::dec << ", count " << m_cnt << ")" << std::endl;
      gpu->memcpy_from_gpu(m_host_address_dst, m_device_address_src, m_cnt);
      m_stream->record_next_done();
      break;
    case stream_memcpy_device_to_device:
      STREAM_DEBUG << "memcpy device-to-device (device addr 0x" << std::hex <<
          m_device_address_src << ", device addr 0x" << m_device_address_dst <<
          std::dec << ", count " << m_cnt << ")" << std::endl;
      gpu->memcpy_gpu_to_gpu(m_device_address_dst, m_device_address_src, m_cnt);
      m_stream->record_next_done();
      break;
    case stream_memcpy_to_symbol:
      STREAM_DEBUG << "memcpy to symbol (host addr 0x" << std::hex << m_host_address_src <<
          std::dec << ", count " << m_cnt << ")" << std::endl;
      gpgpu_ptx_sim_memcpy_symbol(m_symbol, m_host_address_src, m_cnt, m_offset,
                                  1, gpu);
      m_stream->record_next_done();
      break;
    case stream_memcpy_from_symbol:
      STREAM_DEBUG << "memcpy from symbol (host addr 0x" << std::hex << m_host_address_dst <<
          std::dec << ", count " << m_cnt << ")" << std::endl;
      gpgpu_ptx_sim_memcpy_symbol(m_symbol, m_host_address_dst, m_cnt, m_offset,
                                  0, gpu);
      m_stream->record_next_done();
      break;
    case stream_kernel_launch:
      if (gpu->can_start_kernel()) {
        gpu->set_cache_config(m_kernel->name());
        STREAM_DEBUG << "kernel '" << m_kernel->name() << "' transfer to GPU hardware scheduler" <<
            std::endl;
        if (m_sim_mode) {
          gpgpu_cuda_ptx_sim_main_func(*m_kernel);
        } else {
          STREAM_DEBUG << "Launching GPGPU-sim for kernel '" << m_kernel->name() << "'" <<
              std::endl;
          gpu->launch(m_kernel);
        }
      }
      break;
    case stream_event:
      {
        STREAM_DEBUG << "event update" << std::endl;
        // TODO may need to tell gpu-sim::cycle() to wait until the next stream
        // event before continuing
        time_t wallclock = time(NULL);
        m_event->update(gpu_tot_sim_cycle, wallclock);
        m_stream->record_next_done();
        break;
      }
    default:
      assert(false);
  }
  m_done = true;
}

std::ostream& operator<<(std::ostream& o, const stream_operation& s) {
  o << " stream operation ";
  switch (s.m_type) {
    case stream_event:
      o << "event";
      break;
    case stream_kernel_launch:
      o << "kernel";
      break;
    case stream_memcpy_device_to_device:
      o << "memcpy device-to-device";
      break;
    case stream_memcpy_device_to_host:
      o << "memcpy device-to-host";
      break;
    case stream_memcpy_host_to_device:
      o << "memcpy host-to-device";
      break;
    case stream_memcpy_to_symbol:
      o << "memcpy to symbol";
      break;
    case stream_memcpy_from_symbol:
      o << "memcpy from symbol";
      break;
    case stream_no_op:
      o << "no-op";
      break;
    default:
      assert(false);
  }
  return o;
}

stream_manager::stream_manager(gpgpu_sim *gpu, bool cuda_launch_blocking) {
  m_gpu = gpu;
  m_cuda_launch_blocking = cuda_launch_blocking;
  pthread_mutex_init(&m_lock, NULL);
}

bool stream_manager::operation(bool *sim) {
  pthread_mutex_lock(&m_lock);
  bool check = check_finished_kernel();
  if (check) m_gpu->print_stats();
  stream_operation op = front();
  op.do_operation(m_gpu);
  pthread_mutex_unlock(&m_lock);
  // pthread_mutex_lock(&m_lock);
  // simulate a clock cycle on the GPU
  return check;
}

bool stream_manager::check_finished_kernel() {
  unsigned grid_uid = m_gpu->finished_kernel();
  bool check = register_finished_kernel(grid_uid);
  return check;
}

bool stream_manager::register_finished_kernel(unsigned grid_uid) {
  // called by gpu simulation thread
  if (grid_uid > 0) {
    CUstream_st *stream = m_grid_id_to_stream[grid_uid];
    kernel_info_t *kernel = stream->front().get_kernel();
    assert(grid_uid == kernel->get_uid());
    stream->record_next_done();
    m_grid_id_to_stream.erase(grid_uid);
    delete kernel;
    return true;
  } else {
    return false;
  }
  return false;
}

stream_operation stream_manager::front() {
  // called by gpu simulation thread
  stream_operation result;
  std::list<struct CUstream_st *>::iterator s;
  for (s = m_streams.begin(); s != m_streams.end(); s++) {
    CUstream_st *stream = *s;
    if (!stream->busy() && !stream->empty()) {
      result = stream->next();
      if (result.is_kernel()) {
        unsigned grid_id = result.get_kernel()->get_uid();
        result.get_kernel()->set_stream_id(stream->get_stream_id());
        result.get_kernel()->set_done_id(stream->getoncedone());
        m_grid_id_to_stream[grid_id] = stream;
      }
      break;
    }
  }
  return result;
}

void stream_manager::add_stream(struct CUstream_st *stream) {
  // called by host thread
  pthread_mutex_lock(&m_lock);
  m_streams.push_back(stream);
  pthread_mutex_unlock(&m_lock);
}

void stream_manager::destroy_stream(CUstream_st *stream) {
  // called by host thread
  pthread_mutex_lock(&m_lock);
  while (!stream->empty())
    ;
  std::list<CUstream_st *>::iterator s;
  for (s = m_streams.begin(); s != m_streams.end(); s++) {
    if (*s == stream) {
      m_streams.erase(s);
      break;
    }
  }
  delete stream;
  pthread_mutex_unlock(&m_lock);
}

bool stream_manager::concurrent_streams_empty() {
  bool result = true;
  // called by gpu simulation thread
  std::list<struct CUstream_st *>::iterator s;
  pthread_mutex_lock(&m_lock);
  for (s = m_streams.begin(); s != m_streams.end(); ++s) {
    struct CUstream_st *stream = *s;
    if (!stream->empty()) {
      result = false;
    }
  }
  pthread_mutex_unlock(&m_lock);
  return result;
}

bool stream_manager::empty() { return concurrent_streams_empty(); }

std::ostream& operator<<(std::ostream& o, const stream_manager& s) {
  pthread_mutex_lock(&s.m_lock);
  o << "GPGPU-Sim API: Stream Manager State" << std::endl;
  for (std::list<struct CUstream_st*>::const_iterator it = s.m_streams.cbegin();
      it != s.m_streams.cend(); it++) {
    if (!(*it)->empty()) {
      o << *it << std::endl;
    }
  }
  pthread_mutex_unlock(&s.m_lock);
  return o;
}

void stream_manager::push(stream_operation op) {
  struct CUstream_st *stream = op.get_stream();
  assert(stream && "Launched operation on null stream?????");
  // block if stream 0 (or concurrency disabled) and pending concurrent
  // operations exist
  if (m_cuda_launch_blocking)
    while (!concurrent_streams_empty())
      ;  // spin waiting for empty

  pthread_mutex_lock(&m_lock);
  stream->push(op);

  STREAM_DEBUG << *this;
  pthread_mutex_unlock(&m_lock);

  if (m_cuda_launch_blocking) {
    unsigned int wait_amount = 100;
    unsigned int wait_cap = 100000;  // 100ms
    while (!empty()) {
      // sleep to prevent CPU hog by empty spin
      usleep(wait_amount);
      wait_amount *= 2;
      if (wait_amount > wait_cap) wait_amount = wait_cap;
    }
  }
}
