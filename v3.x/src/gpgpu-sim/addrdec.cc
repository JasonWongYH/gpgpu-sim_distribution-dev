// Copyright (c) 2009-2011, Wilson W.L. Fung, Tor M. Aamodt, Ali Bakhoda,
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

#include <cstdint>
#include <string>

#include "../cuda-sim/cuda-sim.h"
#include "../option_parser.h"
#include "../tr1_hash_map.h"
#include "addrdec.h"
#include "ConfigOptions.h"
#include "dram.h"
#include "gpu-sim.h"


#define DEBUG_ENABLE 0
#define ALLOC_DEBUG \
  if (DEBUG_ENABLE) \
  std::cout << "ALLOC DEBUG: "
#define MERGE_DEBUG \
  if (DEBUG_ENABLE) \
  std::cout << "MERGE DEBUG: "

static long int powli(long int x, long int y);
static unsigned int LOGB2_32(unsigned int v);
static new_addr_type addrdec_packbits(new_addr_type mask,
                                      new_addr_type val,
                                      unsigned char high,
                                      unsigned char low);
static void addrdec_getmasklimit(new_addr_type mask,
                                 unsigned char* high,
                                 unsigned char* low);

linear_to_raw_address_translation::linear_to_raw_address_translation() {
  addrdec_option = NULL;
  ADDR_CHIP_S = 10;
  memset(addrdec_mklow, 0, N_ADDRDEC);
  memset(addrdec_mkhigh, 64, N_ADDRDEC);
  addrdec_mask[0] = 0x0000000000001C00;
  addrdec_mask[1] = 0x0000000000000300;
  addrdec_mask[2] = 0x000000000FFF0000;
  addrdec_mask[3] = 0x000000000000E0FF;
  addrdec_mask[4] = 0x000000000000000F;
}

void linear_to_raw_address_translation::addrdec_setoption(option_parser_t opp) {
  option_parser_register(opp, "-gpgpu_mem_addr_mapping", OPT_CSTR,
                         &addrdec_option,
                         "mapping memory address to dram model {dramid@<start "
                         "bit>;<memory address map>}",
                         NULL);
  option_parser_register(opp, "-gpgpu_mem_address_mask", OPT_INT32,
                         &gpgpu_mem_address_mask,
                         "0 = old addressing mask, 1 = new addressing mask, 2 "
                         "= new add. mask + flipped bank sel and chip sel bits",
                         "0");
}

void linear_to_raw_address_translation::set_mmu(mmu* mmu) {
  this->m_mmu = mmu;
}

new_addr_type linear_to_raw_address_translation::partition_address(
    new_addr_type in_addr,
    appid_t appID,
    unsigned level,
    bool isRead) const {
  ALLOC_DEBUG
      << "Address decoder want to parse memory partition ID for address = "
      << std::hex << in_addr << std::dec << ", appID = " << appID
      << ", level = " << level << std::endl;

  new_addr_type addr;
  if (level == DRAM_CMD || appID == App::pt_space.appid)
    addr = in_addr;
  else
    addr = m_mmu->get_pa(in_addr, appID, isRead);
  // Check if this address space has been allocated before (in case of new local memory reference)

  if ((uint64_t)addr == UINT64_MAX) {
    MERGE_DEBUG
        << "Trying to decode memory partition before VA is allcated, VA = "
        << std::hex << in_addr << std::dec << ", cycle "
        << gpu_sim_cycle + gpu_tot_sim_cycle
        << ". Cannot translate to a meaningful PA!!!" << std::endl;
  }

  // Do not need to update any metadata here.
  MERGE_DEBUG
      << "Address decoder want to parse memory partition ID for address = "
      << std::hex << in_addr << std::dec << ", appID = " << appID
      << ", level = " << level << ". Got physical address = " << std::hex
      << addr << std::dec << std::endl;

  if (!gap) {
    return addrdec_packbits(~(addrdec_mask[CHIP] | sub_partition_id_mask), addr,
                            64, 0);
  } else {
    // see addrdec_tlx for explanation
    unsigned long long int partition_addr;
    partition_addr = ((addr >> ADDR_CHIP_S) / m_n_channel) << ADDR_CHIP_S;
    partition_addr |= addr & ((1 << ADDR_CHIP_S) - 1);
    // remove the part of address that constributes to the sub partition ID
    partition_addr =
        addrdec_packbits(~sub_partition_id_mask, partition_addr, 64, 0);
    return partition_addr;
  }
}

bool linear_to_raw_address_translation::addrdec_tlx(new_addr_type in_addr,
                                                    addrdec_t* tlx,
                                                    appid_t appID,
                                                    unsigned level,
                                                    bool isRead) const {
  bool fault = false;
  ALLOC_DEBUG << "Address decoder want to parse physical mapping for address = "
              << std::hex << in_addr << std::dec << ", appID = " << appID
              << ", level = " << level << std::endl;
  new_addr_type addr;
  if (level == DRAM_CMD || appID == App::pt_space.appid)
    addr = in_addr;
  else
    addr = m_mmu->get_pa(in_addr, appID, isRead);  // Check if this address
                                                   // space has been allocated
                                                   // before (in case of new
                                                   // local memory reference)
  MERGE_DEBUG << "Decoing in addrdec_tlx, VA = " << std::hex << in_addr
              << std::dec << ". Got = " << std::hex << addr << std::dec
              << ", cycle = " << gpu_sim_cycle + gpu_tot_sim_cycle
              << ". Calling translate(" << appID << ","
              << (((uint64_t)App::get_app(appID)->addr_offset) << 48 | in_addr)
              << ")" << std::endl;

  if ((uint64_t)addr == UINT64_MAX) {
    MERGE_DEBUG << "Trying to decode PA before VA is allcated, VA = "
                << std::hex << in_addr << std::dec << ", cycle "
                << gpu_sim_cycle + gpu_tot_sim_cycle
                << ". Cannot translate to a meaningful PA!!!" << std::endl;
  }

  // TODO: Update the metadata for this address
  MERGE_DEBUG
      << "Address decoder want to parse physical address ID for address = "
      << std::hex << in_addr << std::dec << ", appID = " << appID
      << ", level = " << level << ". Got physical address = " << std::hex
      << addr << std::dec << std::endl;

  unsigned long long int addr_for_chip, rest_of_addr;
  if (!gap) {
    tlx->chip = addrdec_packbits(addrdec_mask[CHIP], addr, addrdec_mkhigh[CHIP],
                                 addrdec_mklow[CHIP]);
    tlx->bk = addrdec_packbits(addrdec_mask[BK], addr, addrdec_mkhigh[BK],
                               addrdec_mklow[BK]);
    tlx->row = addrdec_packbits(addrdec_mask[ROW], addr, addrdec_mkhigh[ROW],
                                addrdec_mklow[ROW]);
    tlx->col = addrdec_packbits(addrdec_mask[COL], addr, addrdec_mkhigh[COL],
                                addrdec_mklow[COL]);
    tlx->burst = addrdec_packbits(addrdec_mask[BURST], addr,
                                  addrdec_mkhigh[BURST], addrdec_mklow[BURST]);
  } else {
    // Split the given address at ADDR_CHIP_S into (MSBs,LSBs)
    // - extract chip address using modulus of MSBs
    // - recreate the rest of the address by stitching the quotient of MSBs and
    // the LSBs
    addr_for_chip = (addr >> ADDR_CHIP_S) % m_n_channel;
    rest_of_addr = ((addr >> ADDR_CHIP_S) / m_n_channel) << ADDR_CHIP_S;
    rest_of_addr |= addr & ((1 << ADDR_CHIP_S) - 1);
    tlx->chip = addr_for_chip;
    tlx->bk = addrdec_packbits(addrdec_mask[BK], rest_of_addr,
                               addrdec_mkhigh[BK], addrdec_mklow[BK]);
    tlx->row = addrdec_packbits(addrdec_mask[ROW], rest_of_addr,
                                addrdec_mkhigh[ROW], addrdec_mklow[ROW]);
    tlx->col = addrdec_packbits(addrdec_mask[COL], rest_of_addr,
                                addrdec_mkhigh[COL], addrdec_mklow[COL]);
    tlx->burst = addrdec_packbits(addrdec_mask[BURST], rest_of_addr,
                                  addrdec_mkhigh[BURST], addrdec_mklow[BURST]);
  }

  // combine the chip address and the lower bits of DRAM bank address to form
  // the subpartition ID
  unsigned sub_partition_addr_mask = m_n_sub_partition_in_channel - 1;
  tlx->sub_partition = tlx->chip * m_n_sub_partition_in_channel +
                       (tlx->bk & sub_partition_addr_mask);

  // set subarray
  tlx->subarray = addr & m_subarray_bits;

  return fault;
}

void linear_to_raw_address_translation::addrdec_parseoption(
    const char* option) {
  unsigned int dramid_start = 0;
  int dramid_parsed = sscanf(option, "dramid@%d", &dramid_start);
  if (dramid_parsed == 1) {
    ADDR_CHIP_S = dramid_start;
  } else {
    ADDR_CHIP_S = -1;
  }

  const char* cmapping = strchr(option, ';');
  if (cmapping == NULL) {
    cmapping = option;
  } else {
    cmapping += 1;
  }

  addrdec_mask[CHIP] = 0x0;
  addrdec_mask[BK] = 0x0;
  addrdec_mask[ROW] = 0x0;
  addrdec_mask[COL] = 0x0;
  addrdec_mask[BURST] = 0x0;

  int ofs = 63;
  while ((*cmapping) != '\0') {
    switch (*cmapping) {
      case 'D':
      case 'd':
        assert(dramid_parsed != 1);
        addrdec_mask[CHIP] |= (1ULL << ofs);
        ofs--;
        break;
      case 'A':
      case 'a':
        addrdec_mask[SUBARRAY] |= (1ULL << ofs);
        ofs--;
        break;
      case 'B':
      case 'b':
        addrdec_mask[BK] |= (1ULL << ofs);
        ofs--;
        break;
      case 'R':
      case 'r':
        addrdec_mask[ROW] |= (1ULL << ofs);
        ofs--;
        break;
      case 'C':
      case 'c':
        addrdec_mask[COL] |= (1ULL << ofs);
        ofs--;
        break;
      case 'S':
      case 's':
        addrdec_mask[BURST] |= (1ULL << ofs);
        addrdec_mask[COL] |= (1ULL << ofs);
        ofs--;
        break;
      // ignore bit
      case '0':
        ofs--;
        break;
      // ignore character
      case '|':
      case ' ':
      case '.':
        break;
      default:
        std::cerr << "ERROR: Invalid address mapping character '" << *cmapping
                  << "' in option '" << option << "'" << std::endl;
        assert(false);
    }
    cmapping += 1;
  }

  if (ofs != -1) {
    std::cerr << "ERROR: Invalid address mapping length (" << 63 - ofs
              << ") in option '" << option << "'" << std::endl;
    assert(false);
  }
}

void linear_to_raw_address_translation::init(
    unsigned int n_channel,
    unsigned int n_sub_partition_in_channel,
    memory_config* config) {
  m_config = config;
  m_n_bk = m_config->nbk;
  m_n_bkgrp = m_config->nbkgrp;

  unsigned i;
  unsigned long long int mask;
  unsigned int nchipbits = LOGB2_32(n_channel);
  m_n_channel = n_channel;
  m_n_sub_partition_in_channel = n_sub_partition_in_channel;

  gap = (n_channel - ::powli(2, nchipbits));
  if (gap) {
    nchipbits++;
  }
  switch (gpgpu_mem_address_mask) {
    case 0:
      // old, added 2row bits, use #define ADDR_CHIP_S 10
      ADDR_CHIP_S = 10;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000000300;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x0000000000001CFF;
      break;
    case 1:
      ADDR_CHIP_S = 13;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 2:
      ADDR_CHIP_S = 11;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 3:
      ADDR_CHIP_S = 11;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x000000000FFFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 14:
      ADDR_CHIP_S = 14;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 15:
      ADDR_CHIP_S = 15;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 16:
      ADDR_CHIP_S = 16;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 6:
      ADDR_CHIP_S = 6;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 5:
      ADDR_CHIP_S = 5;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 100:
      ADDR_CHIP_S = 1;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000000003;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x0000000000001FFC;
      break;
    case 103:
      ADDR_CHIP_S = 3;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000000003;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x0000000000001FFC;
      break;
    case 106:
      ADDR_CHIP_S = 6;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000001800;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x00000000000007FF;
      break;
    case 160:
      // old, added 2row bits, use #define ADDR_CHIP_S 10
      ADDR_CHIP_S = 6;
      addrdec_mask[CHIP] = 0x0000000000000000;
      addrdec_mask[BK] = 0x0000000000000300;
      addrdec_mask[ROW] = 0x0000000007FFE000;
      addrdec_mask[COL] = 0x0000000000001CFF;
      break;
    default:
      break;
  }

  if (addrdec_option != NULL)
    addrdec_parseoption(addrdec_option);

  m_subarray_bits = addrdec_mask[SUBARRAY];

  if (ADDR_CHIP_S != -1) {
    if (!gap) {
      // number of chip is power of two:
      // - insert CHIP mask starting at the bit position ADDR_CHIP_S
      mask = ((unsigned long long int)1 << ADDR_CHIP_S) - 1;
      addrdec_mask[BK] =
          ((addrdec_mask[BK] & ~mask) << nchipbits) | (addrdec_mask[BK] & mask);
      addrdec_mask[ROW] = ((addrdec_mask[ROW] & ~mask) << nchipbits) |
                          (addrdec_mask[ROW] & mask);
      addrdec_mask[COL] = ((addrdec_mask[COL] & ~mask) << nchipbits) |
                          (addrdec_mask[COL] & mask);

      for (i = ADDR_CHIP_S; i < (ADDR_CHIP_S + nchipbits); i++) {
        mask = (unsigned long long int)1 << i;
        addrdec_mask[CHIP] |= mask;
      }
    }  // otherwise, no need to change the masks
  } else {
    // make sure n_channel is power of two when explicit dram id mask is used
    assert((n_channel & (n_channel - 1)) == 0);
  }
  // make sure m_n_sub_partition_in_channel is power of two
  assert((m_n_sub_partition_in_channel & (m_n_sub_partition_in_channel - 1)) ==
         0);

  addrdec_getmasklimit(addrdec_mask[CHIP], &addrdec_mkhigh[CHIP],
                       &addrdec_mklow[CHIP]);
  addrdec_getmasklimit(addrdec_mask[BK], &addrdec_mkhigh[BK],
                       &addrdec_mklow[BK]);
  addrdec_getmasklimit(addrdec_mask[ROW], &addrdec_mkhigh[ROW],
                       &addrdec_mklow[ROW]);
  addrdec_getmasklimit(addrdec_mask[COL], &addrdec_mkhigh[COL],
                       &addrdec_mklow[COL]);
  addrdec_getmasklimit(addrdec_mask[BURST], &addrdec_mkhigh[BURST],
                       &addrdec_mklow[BURST]);
  addrdec_getmasklimit(addrdec_mask[SUBARRAY], &addrdec_mkhigh[SUBARRAY],
                       &addrdec_mklow[SUBARRAY]);

  std::cout << "addr_dec_mask[CHIP]  = " << std::hex << addrdec_mask[CHIP]
            << std::dec << " \thigh:" << addrdec_mkhigh[CHIP]
            << " low:" << addrdec_mklow[CHIP] << std::endl;
  std::cout << "addr_dec_mask[BK]    = " << std::hex << addrdec_mask[BK]
            << std::dec << " \thigh:" << addrdec_mkhigh[BK]
            << " low:" << addrdec_mklow[BK] << std::endl;
  std::cout << "addr_dec_mask[ROW]   = " << std::hex << addrdec_mask[ROW]
            << std::dec << " \thigh:" << addrdec_mkhigh[ROW]
            << " low:" << addrdec_mklow[ROW] << std::endl;
  std::cout << "addr_dec_mask[COL]   = " << std::hex << addrdec_mask[COL]
            << std::dec << " \thigh:" << addrdec_mkhigh[COL]
            << " low:" << addrdec_mklow[COL] << std::endl;
  std::cout << "addr_dec_mask[BURST] = " << std::hex << addrdec_mask[BURST]
            << std::dec << " \thigh:" << addrdec_mkhigh[BURST]
            << " low:" << addrdec_mklow[BURST] << std::endl;
  std::cout << "addr_dec_mask[SUBARRAY] = " << std::hex
            << addrdec_mask[SUBARRAY] << std::dec
            << " \thigh:" << addrdec_mkhigh[SUBARRAY]
            << " low:" << addrdec_mklow[SUBARRAY] << std::endl;

  // create the sub partition ID mask (for removing the sub partition ID from
  // the partition address)
  sub_partition_id_mask = 0;
  if (m_n_sub_partition_in_channel > 1) {
    unsigned n_sub_partition_log2 = LOGB2_32(m_n_sub_partition_in_channel);
    unsigned pos = 0;
    for (unsigned i = addrdec_mklow[BK]; i < addrdec_mkhigh[BK]; i++) {
      if ((addrdec_mask[BK] & ((unsigned long long int)1 << i)) != 0) {
        sub_partition_id_mask |= ((unsigned long long int)1 << i);
        pos++;
        if (pos >= n_sub_partition_log2)
          break;
      }
    }
  }
  std::cout << "sub_partition_id_mask = " << std::hex << sub_partition_id_mask
            << std::dec << std::endl;
}

bool operator==(const addrdec_t& x, const addrdec_t& y) {
  return (memcmp(&x, &y, sizeof(addrdec_t)) == 0);
}

bool operator<(const addrdec_t& x, const addrdec_t& y) {
  if (x.chip >= y.chip)
    return false;
  else if (x.bk >= y.bk)
    return false;
  else if (x.row >= y.row)
    return false;
  else if (x.col >= y.col)
    return false;
  else if (x.burst >= y.burst)
    return false;
  else
    return true;
}

class hash_addrdec_t {
 public:
  size_t operator()(const addrdec_t& x) const {
    return (x.chip ^ x.bk ^ x.row ^ x.col ^ x.burst);
  }
};

std::ostream& operator<<(std::ostream& o, const addrdec_t& a) {
  o << "\tchip:" << std::hex << a.chip;
  o << "\trow:" << a.row;
  o << "\tcol:" << a.col;
  o << "\tbk:" << a.bk;
  o << "\tburst:" << a.burst;
  o << "\tsub_partition:" << a.sub_partition << std::dec << " ";
  return o;
}

static long int powli(long int x, long int y)  // compute x to the y
{
  long int r = 1;
  int i;
  for (i = 0; i < y; ++i) {
    r *= x;
  }
  return r;
}

static unsigned int LOGB2_32(unsigned int v) {
  unsigned int shift;
  unsigned int r;

  r = 0;

  shift = ((v & 0xFFFF0000) != 0) << 4;
  v >>= shift;
  r |= shift;
  shift = ((v & 0xFF00) != 0) << 3;
  v >>= shift;
  r |= shift;
  shift = ((v & 0xF0) != 0) << 2;
  v >>= shift;
  r |= shift;
  shift = ((v & 0xC) != 0) << 1;
  v >>= shift;
  r |= shift;
  shift = ((v & 0x2) != 0) << 0;
  v >>= shift;
  r |= shift;

  return r;
}

static new_addr_type addrdec_packbits(new_addr_type mask,
                                      new_addr_type val,
                                      unsigned char high,
                                      unsigned char low) {
  unsigned pos = 0;
  new_addr_type result = 0;
  for (unsigned i = low; i < high; i++) {
    if ((mask & ((unsigned long long int)1 << i)) != 0) {
      result |= ((val & ((unsigned long long int)1 << i)) >> i) << pos;
      pos++;
    }
  }
  return result;
}

static void addrdec_getmasklimit(new_addr_type mask,
                                 unsigned char* high,
                                 unsigned char* low) {
  *high = 64;
  *low = 0;
  int i;
  int low_found = 0;

  for (i = 0; i < 64; i++) {
    if ((mask & ((unsigned long long int)1 << i)) != 0) {
      if (low_found) {
        *high = i + 1;
      } else {
        *high = i + 1;
        *low = i;
        low_found = 1;
      }
    }
  }
}
