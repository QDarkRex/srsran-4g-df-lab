/**
 * Copyright 2013-2023 Software Radio Systems Limited
 *
 * This file is part of srsRAN.
 *
 * srsRAN is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * srsRAN is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * A copy of the GNU Affero General Public License can be found in
 * the LICENSE file in the top-level directory of this distribution
 * and at http://www.gnu.org/licenses/.
 *
 */

#include <complex.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <arpa/inet.h>
#include <pthread.h>

#include "srsran/config.h"
#include "srsran/phy/ch_estimation/cedron_freq_estimator.h"
#include "srsran/phy/ch_estimation/chest_ul.h"
#include "srsran/phy/dft/dft_precoding.h"
#include "srsran/phy/utils/convolution.h"
#include "srsran/phy/utils/vector.h"
#include "srsran/srsran.h"

#define NOF_REFS_SYM (q->cell.nof_prb * SRSRAN_NRE)
#define NOF_REFS_SF (NOF_REFS_SYM * 2) // 2 reference symbols per subframe

#define MAX_REFS_SYM (max_prb * SRSRAN_NRE)
#define MAX_REFS_SF (max_prb * SRSRAN_NRE * 2) // 2 reference symbols per subframe

/** 3GPP LTE Downlink channel estimator and equalizer.
 * Estimates the channel in the resource elements transmitting references and interpolates for the rest
 * of the resource grid.
 *
 * The equalizer uses the channel estimates to produce an estimation of the transmitted symbol.
 *
 * This object depends on the srsran_refsignal_t object for creating the LTE CSR signal.
 */

int srsran_chest_ul_init(srsran_chest_ul_t* q, uint32_t max_prb)
{
  int ret = SRSRAN_ERROR_INVALID_INPUTS;
  if (q != NULL) {
    bzero(q, sizeof(srsran_chest_ul_t));

    q->tmp_noise = srsran_vec_cf_malloc(MAX_REFS_SF);
    if (!q->tmp_noise) {
      perror("malloc");
      goto clean_exit;
    }
    q->pilot_estimates = srsran_vec_cf_malloc(MAX_REFS_SF);
    if (!q->pilot_estimates) {
      perror("malloc");
      goto clean_exit;
    }
    for (int i = 0; i < 4; i++) {
      q->pilot_estimates_tmp[i] = srsran_vec_cf_malloc(MAX_REFS_SF);
      if (!q->pilot_estimates_tmp[i]) {
        perror("malloc");
        goto clean_exit;
      }
    }
    q->pilot_recv_signal = srsran_vec_cf_malloc(MAX_REFS_SF + 1);
    if (!q->pilot_recv_signal) {
      perror("malloc");
      goto clean_exit;
    }

    q->pilot_known_signal = srsran_vec_cf_malloc(MAX_REFS_SF + 1);
    if (!q->pilot_known_signal) {
      perror("malloc");
      goto clean_exit;
    }

    if (srsran_interp_linear_vector_init(&q->srsran_interp_linvec, MAX_REFS_SYM)) {
      ERROR("Error initializing vector interpolator");
      goto clean_exit;
    }

    q->smooth_filter_len = 3;
    srsran_chest_set_smooth_filter3_coeff(q->smooth_filter, 0.3333);

    q->dmrs_signal_configured = false;

    if (srsran_refsignal_dmrs_pusch_pregen_init(&q->dmrs_pregen, max_prb)) {
      ERROR("Error allocating memory for pregenerated signals");
      goto clean_exit;
    }

    if (srsran_cedron_freq_est_init(&q->srsran_cedron_freq_est, max_prb)) {
      ERROR("Error initializing cedron freq estimation algorithm.");
      goto clean_exit;
    }
  }

  ret = SRSRAN_SUCCESS;

clean_exit:
  if (ret != SRSRAN_SUCCESS) {
    srsran_chest_ul_free(q);
  }
  return ret;
}

void srsran_chest_ul_free(srsran_chest_ul_t* q)
{
  srsran_refsignal_dmrs_pusch_pregen_free(&q->dmrs_signal, &q->dmrs_pregen);

  if (q->tmp_noise) {
    free(q->tmp_noise);
  }
  srsran_interp_linear_vector_free(&q->srsran_interp_linvec);
  srsran_cedron_freq_est_free(&q->srsran_cedron_freq_est);

  if (q->pilot_estimates) {
    free(q->pilot_estimates);
  }
  for (int i = 0; i < 4; i++) {
    if (q->pilot_estimates_tmp[i]) {
      free(q->pilot_estimates_tmp[i]);
    }
  }
  if (q->pilot_recv_signal) {
    free(q->pilot_recv_signal);
  }
  if (q->pilot_known_signal) {
    free(q->pilot_known_signal);
  }
  bzero(q, sizeof(srsran_chest_ul_t));
}

int srsran_chest_ul_res_init(srsran_chest_ul_res_t* q, uint32_t max_prb)
{
  bzero(q, sizeof(srsran_chest_ul_res_t));
  q->antenna_idx = 0; // Set a default
  q->nof_re = SRSRAN_SF_LEN_RE(max_prb, SRSRAN_CP_NORM);
  q->ce     = srsran_vec_cf_malloc(q->nof_re);
  if (!q->ce) {
    perror("malloc");
    return -1;
  }
  return 0;
}

void srsran_chest_ul_res_set_identity(srsran_chest_ul_res_t* q)
{
  for (uint32_t i = 0; i < q->nof_re; i++) {
    q->ce[i] = 1.0;
  }
}

void srsran_chest_ul_res_free(srsran_chest_ul_res_t* q)
{
  if (q->ce) {
    free(q->ce);
  }
}

int srsran_chest_ul_set_cell(srsran_chest_ul_t* q, srsran_cell_t cell)
{
  int ret = SRSRAN_ERROR_INVALID_INPUTS;
  if (q != NULL && srsran_cell_isvalid(&cell)) {
    if (cell.id != q->cell.id || q->cell.nof_prb == 0) {
      q->cell = cell;
      ret     = srsran_refsignal_ul_set_cell(&q->dmrs_signal, cell);
      if (ret != SRSRAN_SUCCESS) {
        ERROR("Error initializing CSR signal (%d)", ret);
        return SRSRAN_ERROR;
      }

      if (srsran_interp_linear_vector_resize(&q->srsran_interp_linvec, NOF_REFS_SYM)) {
        ERROR("Error initializing vector interpolator");
        return SRSRAN_ERROR;
      }
    }
    ret = SRSRAN_SUCCESS;
  }
  return ret;
}

void srsran_chest_ul_pregen(srsran_chest_ul_t*                 q,
                            srsran_refsignal_dmrs_pusch_cfg_t* cfg,
                            srsran_refsignal_srs_cfg_t*        srs_cfg)
{
  srsran_refsignal_dmrs_pusch_pregen(&q->dmrs_signal, &q->dmrs_pregen, cfg);
  q->dmrs_signal_configured = true;

  if (srs_cfg) {
    srsran_refsignal_srs_pregen(&q->dmrs_signal, &q->srs_pregen, srs_cfg, cfg);
    q->srs_signal_configured = true;
  }
}

/* Uses the difference between the averaged and non-averaged pilot estimates */
static float estimate_noise_pilots(srsran_chest_ul_t* q, cf_t* ce, uint32_t nslots, uint32_t nrefs, uint32_t n_prb[2])
{
  float power = 0;
  for (int i = 0; i < nslots; i++) {
    power += srsran_chest_estimate_noise_pilots(
        &q->pilot_estimates[i * nrefs],
        &ce[SRSRAN_REFSIGNAL_UL_L(i, q->cell.cp) * q->cell.nof_prb * SRSRAN_NRE + n_prb[i] * SRSRAN_NRE],
        q->tmp_noise,
        nrefs);
  }

  power /= nslots;

  if (q->smooth_filter_len == 3) {
    // Calibrated for filter length 3
    float w = q->smooth_filter[0];
    float a = 7.419 * w * w + 0.1117 * w - 0.005387;
    return (power / (a * 0.8));
  } else {
    return power;
  }
}

// The interpolator currently only supports same frequency allocation for each subframe
#define cesymb(i) ce[SRSRAN_RE_IDX(q->cell.nof_prb, i, n_prb[0] * SRSRAN_NRE)]
static void interpolate_pilots(srsran_chest_ul_t* q, cf_t* ce, uint32_t nslots, uint32_t nrefs, uint32_t n_prb[2])
{
#ifdef DO_LINEAR_INTERPOLATION
  uint32_t L1 = SRSRAN_REFSIGNAL_UL_L(0, q->cell.cp);
  uint32_t L2 = SRSRAN_REFSIGNAL_UL_L(1, q->cell.cp);
  uint32_t NL = 2 * SRSRAN_CP_NSYMB(q->cell.cp);

  /* Interpolate in the time domain between symbols */
  srsran_interp_linear_vector3(
      &q->srsran_interp_linvec, &cesymb(L2), &cesymb(L1), &cesymb(L1), &cesymb(L1 - 1), (L2 - L1), L1, false, nrefs);
  srsran_interp_linear_vector3(
      &q->srsran_interp_linvec, &cesymb(L1), &cesymb(L2), NULL, &cesymb(L1 + 1), (L2 - L1), (L2 - L1) - 1, true, nrefs);
  srsran_interp_linear_vector3(&q->srsran_interp_linvec,
                               &cesymb(L1),
                               &cesymb(L2),
                               &cesymb(L2),
                               &cesymb(L2 + 1),
                               (L2 - L1),
                               (NL - L2) - 1,
                               true,
                               nrefs);
#else
  // Instead of a linear interpolation, we just copy the estimates to all symbols in that subframe
  for (int s = 0; s < nslots; s++) {
    for (int i = 0; i < SRSRAN_CP_NSYMB(q->cell.cp); i++) {
      int src_symb = SRSRAN_REFSIGNAL_UL_L(s, q->cell.cp);
      int dst_symb = i + s * SRSRAN_CP_NSYMB(q->cell.cp);

      // skip the symbol with the estimates
      if (dst_symb != src_symb) {
        srsran_vec_cf_copy(&ce[(dst_symb * q->cell.nof_prb + n_prb[s]) * SRSRAN_NRE],
                           &ce[(src_symb * q->cell.nof_prb + n_prb[s]) * SRSRAN_NRE],
                           nrefs);
      }
    }
  }
#endif
}

static void
average_pilots(srsran_chest_ul_t* q, cf_t* input, cf_t* ce, uint32_t nslots, uint32_t nrefs, uint32_t n_prb[2])
{
  for (uint32_t i = 0; i < nslots; i++) {
    srsran_chest_average_pilots(
        &input[i * nrefs],
        &ce[SRSRAN_REFSIGNAL_UL_L(i, q->cell.cp) * q->cell.nof_prb * SRSRAN_NRE + n_prb[i] * SRSRAN_NRE],
        q->smooth_filter,
        nrefs,
        1,
        q->smooth_filter_len);
  }
}

#define MAX_ACTIVE_UES 16
typedef struct {
    char imsi[32];
    uint16_t rnti;
    float spatial_delta;
    float magnitude;
    float multipath_indicator;
    uint32_t last_updated_frame;
    bool active;
    
    // Per-UE antenna caching to prevent crossover
    cf_t     ant0_ce[SRSRAN_MAX_PRB * SRSRAN_NRE]; // full-RB channel estimate from
                                                    // antenna 0, cached for coherent
                                                    // cross-correlation against antenna 1
    uint32_t ant0_nrefs;
    bool ant0_captured;
    float last_avg_m_ant0;
    float last_csi_var_ant0;
    float last_min_m_ant0;
    float last_max_m_ant0;
} active_ue_t;

static active_ue_t active_ues[MAX_ACTIVE_UES] = {0};
// Guards active_ues[] + csi_fp/csi_global_idx/csi_throttle below: chest_ul_estimate()
// runs concurrently across PHY worker threads (enb.conf nof_phy_threads, default 3),
// and these were plain file-scope statics with no synchronization.
static pthread_mutex_t df_state_mutex = PTHREAD_MUTEX_INITIALIZER;

static int compare_active_ue(const void* a, const void* b) {
    const active_ue_t* ue_a = (const active_ue_t*)a;
    const active_ue_t* ue_b = (const active_ue_t*)b;

    if (ue_a->active && !ue_b->active) return -1;
    if (!ue_a->active && ue_b->active) return 1;
    if (!ue_a->active && !ue_b->active) return 0;

    bool has_imsi_a = (strncmp(ue_a->imsi, "RNTI-", 5) != 0);
    bool has_imsi_b = (strncmp(ue_b->imsi, "RNTI-", 5) != 0);

    if (has_imsi_a && !has_imsi_b) return -1;
    if (!has_imsi_a && has_imsi_b) return 1;

    int cmp = strcmp(ue_a->imsi, ue_b->imsi);
    if (cmp != 0) {
        return cmp;
    }
    if (ue_a->last_updated_frame > ue_b->last_updated_frame) return -1;
    if (ue_a->last_updated_frame < ue_b->last_updated_frame) return 1;
    return 0;
}

// Static per-array phase offset (radians) between the RX0/RX1 chains, measured
// once with scripts/calibrate_df.py and written to /tmp/df_calibration.conf as
// a single float. Cable-length / connector mismatch between the two RX chains
// biases every spatial_delta reading by a constant amount, so this must be
// subtracted before the LEFT/RIGHT threshold means anything.
// Loaded once at process start; restart srsenb after (re)running the
// calibration script for a new offset to take effect.
static float get_df_calibration_offset(void)
{
    static bool  loaded     = false;
    static float offset_rad = 0.0f;
    if (!loaded) {
        FILE* fp = fopen("/tmp/df_calibration.conf", "r");
        if (fp != NULL) {
            if (fscanf(fp, "%f", &offset_rad) != 1) {
                offset_rad = 0.0f;
            }
            fclose(fp);
        }
        loaded = true;
    }
    return offset_rad;
}

static bool resolve_imsi_from_rnti(uint16_t rnti, char* imsi_str, size_t max_len)
{
    uint32_t target_s1ap_id = 0;
    bool found_s1ap = false;

    // 1. Prefer direct RNTI -> IMSI observations decoded from UL NAS at the eNB.
    FILE* fp = fopen("/tmp/rnti_imsi.csv", "r");
    if (fp != NULL) {
        char line[256];
        char latest_imsi[32] = "";
        bool found_imsi = false;
        while (fgets(line, sizeof(line), fp) != NULL) {
            unsigned int r = 0;
            char imsi_val[32] = "";
            if (sscanf(line, "%u,%31s", &r, imsi_val) == 2) {
                if (r == rnti) {
                    snprintf(latest_imsi, sizeof(latest_imsi), "%s", imsi_val);
                    latest_imsi[strcspn(latest_imsi, "\r\n, ")] = 0;
                    found_imsi = true;
                }
            }
        }
        fclose(fp);
        if (found_imsi && strlen(latest_imsi) > 0) {
            snprintf(imsi_str, max_len, "%s", latest_imsi);
            return true;
        }
    }

    // 2. Search /tmp/rnti_s1ap.csv for the latest mapping of rnti -> enb_ue_s1ap_id
    fp = fopen("/tmp/rnti_s1ap.csv", "r");
    if (fp != NULL) {
        char line[256];
        while (fgets(line, sizeof(line), fp) != NULL) {
            unsigned int r = 0;
            unsigned int s1ap_id = 0;
            if (sscanf(line, "%u,%u", &r, &s1ap_id) == 2) {
                if (r == rnti) {
                    target_s1ap_id = s1ap_id;
                    found_s1ap = true;
                }
            }
        }
        fclose(fp);
    }

    if (found_s1ap) {
        // 3. Search /tmp/s1ap_imsi.csv for the latest mapping of enb_ue_s1ap_id -> imsi
        fp = fopen("/tmp/s1ap_imsi.csv", "r");
        if (fp != NULL) {
            char line[256];
            char latest_imsi[32] = "";
            bool found_imsi = false;
            while (fgets(line, sizeof(line), fp) != NULL) {
                unsigned int s1ap_id = 0;
                char imsi_val[32] = "";
                if (sscanf(line, "%u,%31s", &s1ap_id, imsi_val) == 2) {
                    if (s1ap_id == target_s1ap_id) {
                        snprintf(latest_imsi, sizeof(latest_imsi), "%s", imsi_val);
                        latest_imsi[strcspn(latest_imsi, "\r\n, ")] = 0;
                        found_imsi = true;
                    }
                }
            }
            fclose(fp);
            if (found_imsi && strlen(latest_imsi) > 0) {
                snprintf(imsi_str, max_len, "%s", latest_imsi);
                FILE* fp_rnti_imsi = fopen("/tmp/rnti_imsi.csv", "a");
                if (fp_rnti_imsi != NULL) {
                    fprintf(fp_rnti_imsi, "%u,%s\n", rnti, latest_imsi);
                    fclose(fp_rnti_imsi);
                }
                return true;
            }
        }
    }

    return false;
}

static void chest_ul_estimate(srsran_chest_ul_t* q,
                              uint32_t               nslots,
                              uint32_t               nrefs_sym,
                              uint32_t               stride,
                              bool                   meas_ta_en,
                              bool                   use_cedron_alg,
                              bool                   write_estimates,
                              uint32_t               n_prb[SRSRAN_NOF_SLOTS_PER_SF],
                              uint16_t               rnti,
                              srsran_chest_ul_res_t* res)
{
    // 1. STANDARD srsRAN CALCULATIONS (CFO, TA, SNR, RSRP)
    if (nslots == 2) {
        float phase = cargf(srsran_vec_dot_prod_conj_ccc(
            &q->pilot_estimates[0 * nrefs_sym], &q->pilot_estimates[1 * nrefs_sym], nrefs_sym));
        res->cfo_hz = phase / (2.0f * (float)M_PI * 0.0005f);
    } else {
        res->cfo_hz = NAN;
    }

    float ta_err = 0.0f;
    if (meas_ta_en) {
        for (int i = 0; i < nslots; i++) {
            if (use_cedron_alg) {
                ta_err += srsran_cedron_freq_estimate(&q->srsran_cedron_freq_est, 
                          &q->pilot_estimates[i * nrefs_sym], nrefs_sym) / nslots;
            } else {
                ta_err += srsran_vec_estimate_frequency(&q->pilot_estimates[i * nrefs_sym], 
                          nrefs_sym) / nslots;
            }
        }
    }

    if (isnormal(ta_err) && stride > 0) {
        ta_err /= (float)stride;
        ta_err /= 15e3f;
        ta_err *= 1e6f;
        ta_err = roundf(ta_err * 10.0f) / 10.0f;
        res->ta_us = ta_err;
    } else {
        res->ta_us = 0.0f;
    }

    if (res->ce != NULL) {
        if (q->smooth_filter_len > 0) {
            average_pilots(q, q->pilot_estimates, res->ce, nslots, nrefs_sym, n_prb);
            if (write_estimates) interpolate_pilots(q, res->ce, nslots, nrefs_sym, n_prb);
            res->noise_estimate = estimate_noise_pilots(q, res->ce, nslots, nrefs_sym, n_prb);
        } else {
            for (int i = 0; i < nslots; i++) {
                srsran_vec_cf_copy(&res->ce[SRSRAN_REFSIGNAL_UL_L(i, q->cell.cp) * q->cell.nof_prb * SRSRAN_NRE + n_prb[i] * SRSRAN_NRE],
                                   &q->pilot_estimates[i * nrefs_sym], nrefs_sym);
            }
            if (write_estimates) interpolate_pilots(q, res->ce, nslots, nrefs_sym, n_prb);
            res->noise_estimate = 0;
        }
    }

    cf_t corr = srsran_vec_acc_cc(q->pilot_recv_signal, nslots * nrefs_sym) / (nslots * nrefs_sym);
    float rsrp_avg = __real__ corr * __real__ corr + __imag__ corr * __imag__ corr;
    float epre = srsran_vec_avg_power_cf(q->pilot_recv_signal, nslots * nrefs_sym);
    rsrp_avg = SRSRAN_MIN(rsrp_avg, epre);

    if (isnormal(res->noise_estimate)) {
        res->snr = epre / res->noise_estimate;
    } else {
        res->snr = NAN;
    }

    res->epre = epre;
    res->epre_dBfs = srsran_convert_power_to_dB(res->epre);
    res->rsrp = rsrp_avg;
    res->rsrp_dBfs = srsran_convert_power_to_dB(res->rsrp);
    res->snr_db = srsran_convert_power_to_dB(res->snr);
    res->noise_estimate_dbFs = srsran_convert_power_to_dBm(res->noise_estimate);

    // === MASTER IMSI-LOCK DASHBOARD: COMPILATION SAFE (STABLE DF) ===
    static FILE* csi_fp           = NULL;
    static uint32_t csi_global_idx = 0;
    static uint32_t csi_throttle   = 0;

    pthread_mutex_lock(&df_state_mutex);

    if (csi_fp == NULL) {
        csi_fp = fopen("/tmp/csi_capture.bin", "wb");
    }

    if (res->ce != NULL && csi_fp != NULL) {
        csi_global_idx++;

        // Find or insert this RNTI in the active_ues array
        int ue_idx = -1;
        for (int i = 0; i < MAX_ACTIVE_UES; i++) {
            if (active_ues[i].active && active_ues[i].rnti == rnti) {
                ue_idx = i;
                break;
            }
        }
        if (ue_idx == -1) {
            for (int i = 0; i < MAX_ACTIVE_UES; i++) {
                if (!active_ues[i].active) {
                    ue_idx = i;
                    break;
                }
            }
        }
        if (ue_idx == -1) {
            uint32_t oldest_frame = 0xffffffff;
            int oldest_idx = 0;
            for (int i = 0; i < MAX_ACTIVE_UES; i++) {
                if (active_ues[i].last_updated_frame < oldest_frame) {
                    oldest_frame = active_ues[i].last_updated_frame;
                    oldest_idx = i;
                }
            }
            ue_idx = oldest_idx;
        }

        // Initialize active UE entry if newly added or matching
        if (active_ues[ue_idx].rnti != rnti) {
            // Slot is new or being recycled from a different RNTI: drop stale
            // IMSI cache so we don't attribute a previous UE's identity to this one.
            active_ues[ue_idx].imsi[0]       = '\0';
            active_ues[ue_idx].ant0_captured = false;
        }
        active_ues[ue_idx].rnti = rnti;
        active_ues[ue_idx].active = true;
        active_ues[ue_idx].last_updated_frame = csi_global_idx;

        // 1. Binary Capture (Throttled to every 10 updates)
        if (csi_throttle++ % 10 == 0) {
            uint32_t header[2] = {csi_global_idx, nrefs_sym};
            fwrite(header, sizeof(uint32_t), 2, csi_fp);
            for (uint32_t i = 0; i < nrefs_sym; i++) {
                uint32_t p_idx = SRSRAN_REFSIGNAL_UL_L(0, q->cell.cp) * q->cell.nof_prb * SRSRAN_NRE + n_prb[0] * SRSRAN_NRE + i;
                float mag = cabsf(res->ce[p_idx]);
                fwrite(&mag, sizeof(float), 1, csi_fp);
            }
        }

        // 2. DF State Logic (Must run for EVERY antenna call to keep sync)
        uint32_t ref_idx = SRSRAN_REFSIGNAL_UL_L(0, q->cell.cp) * q->cell.nof_prb * SRSRAN_NRE + n_prb[0] * SRSRAN_NRE;

        // Calculate average magnitude of current antenna and its variance (csi_var)
        float sum_m = 0;
        float min_m = 99999.0f;
        float max_m = 0.0f;
        for (uint32_t i = 0; i < nrefs_sym; i++) {
            float m = cabsf(res->ce[ref_idx + i]);
            if (m < min_m) min_m = m;
            if (m > max_m) max_m = m;
            sum_m += m;
        }
        float avg_m = sum_m / nrefs_sym;
        float csi_var = (max_m - min_m) / (avg_m + 0.001f);

        if (res->antenna_idx == 0) {
            // Cache the full-RB complex channel estimate (not just its phase) so
            // antenna 1 can coherently correlate against it below.
            uint32_t cap_n = nrefs_sym;
            if (cap_n > SRSRAN_MAX_PRB * SRSRAN_NRE) {
                cap_n = SRSRAN_MAX_PRB * SRSRAN_NRE;
            }
            memcpy(active_ues[ue_idx].ant0_ce, &res->ce[ref_idx], cap_n * sizeof(cf_t));
            active_ues[ue_idx].ant0_nrefs = cap_n;
            active_ues[ue_idx].ant0_captured = true;
            active_ues[ue_idx].last_avg_m_ant0 = avg_m;
            active_ues[ue_idx].last_csi_var_ant0 = csi_var;
            active_ues[ue_idx].last_min_m_ant0 = min_m;
            active_ues[ue_idx].last_max_m_ant0 = max_m;
        } else if (res->antenna_idx == 1 && active_ues[ue_idx].ant0_captured) {
            // Coherent phase-difference estimate: correlate antenna 1's channel
            // vector against the cached antenna 0 vector across every RE in the
            // RB (angle(sum(ce1[i] * conj(ce0[i])))), instead of reading a single
            // subcarrier's phase. Each RE is implicitly weighted by its own
            // magnitude, so REs sitting in a frequency-selective fade contribute
            // less noise to the result — this is the standard interferometric
            // AoA phase estimator.
            uint32_t corr_n = active_ues[ue_idx].ant0_nrefs;
            if (corr_n > nrefs_sym) {
                corr_n = nrefs_sym;
            }
            cf_t corr_sum = 0.0f;
            for (uint32_t i = 0; i < corr_n; i++) {
                corr_sum += res->ce[ref_idx + i] * conjf(active_ues[ue_idx].ant0_ce[i]);
            }
            float spatial_delta = (corr_n > 0) ? cargf(corr_sum) : 0.0f;

            // Remove the static RX-chain phase offset (0.0 rad until calibrated).
            spatial_delta -= get_df_calibration_offset();
            if (spatial_delta > (float)M_PI) spatial_delta -= 2.0f * (float)M_PI;
            if (spatial_delta < -(float)M_PI) spatial_delta += 2.0f * (float)M_PI;

            active_ues[ue_idx].ant0_captured = false;

            // 3. Console Dashboard (Throttled: Shows joined result once every 10 subframes = 20 slots)
            float joined_mag = (active_ues[ue_idx].last_avg_m_ant0 + avg_m) / 2.0f;
            float joined_csi_var = (active_ues[ue_idx].last_csi_var_ant0 + csi_var) / 2.0f;

            char imsi_str[32];
            bool already_resolved = (active_ues[ue_idx].imsi[0] != '\0') &&
                                     (strncmp(active_ues[ue_idx].imsi, "RNTI-", 5) != 0);
            if (already_resolved) {
                // Already known for this UE: skip the disk lookup entirely.
                // resolve_imsi_from_rnti() opens/scans up to 3 files from /tmp on every
                // call, inside the real-time PHY UL path (~1ms TTI budget) — doing that
                // every subframe for every active UE was the real bottleneck.
                snprintf(imsi_str, sizeof(imsi_str), "%s", active_ues[ue_idx].imsi);
            } else {
                bool resolved = resolve_imsi_from_rnti(rnti, imsi_str, sizeof(imsi_str));
                if (!resolved) {
                    snprintf(imsi_str, sizeof(imsi_str), "RNTI-0x%04x", rnti);
                }
            }

            // Write raw data to /tmp/multipath_<IMSI>.csv
            char log_filename[64];
            snprintf(log_filename, sizeof(log_filename), "/tmp/multipath_%s.csv", imsi_str);
            FILE* mp_log_fp = fopen(log_filename, "a");
            if (mp_log_fp != NULL) {
                fseek(mp_log_fp, 0, SEEK_END);
                long size = ftell(mp_log_fp);
                if (size == 0) {
                    fprintf(mp_log_fp, "IMSI,Ant0_Min,Ant0_Max,Ant0_Avg,Ant0_CsiVar,Ant1_Min,Ant1_Max,Ant1_Avg,Ant1_CsiVar,JoinedCsiVar\n");
                }
                fprintf(mp_log_fp, "%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                        imsi_str,
                        active_ues[ue_idx].last_min_m_ant0, active_ues[ue_idx].last_max_m_ant0, active_ues[ue_idx].last_avg_m_ant0, active_ues[ue_idx].last_csi_var_ant0,
                        min_m, max_m, avg_m, csi_var,
                        joined_csi_var);
                fclose(mp_log_fp);
            }

            if (true) {
                // --- DF Bridge Export (UDP Port 5555) ---
                static int udp_sock = -1;
                static struct sockaddr_in servaddr;
                if (udp_sock == -1) {
                    udp_sock = socket(AF_INET, SOCK_DGRAM, 0);
                    servaddr.sin_family = AF_INET;
                    servaddr.sin_port = htons(5555); // Bridge port
                    servaddr.sin_addr.s_addr = inet_addr("127.0.0.1"); // was "0.0.0.0" — not a valid sendto() destination
                }

                char packet[128];
                // IMSI, Phase Delta, joined Magnitude, and joined CSI variance
                // (multipath/quality indicator - same value the console dashboard
                // uses for its LOW/MID/HIGH label). Consumers can use this to
                // distrust a sample instead of plotting a noisy reading as-is.
                sprintf(packet, "%s,%.4f,%.2f,%.4f", imsi_str, spatial_delta, joined_mag, joined_csi_var);
                sendto(udp_sock, packet, strlen(packet), 0, (const struct sockaddr *)&servaddr, sizeof(servaddr));

                // Update cache entry
                snprintf(active_ues[ue_idx].imsi, sizeof(active_ues[ue_idx].imsi), "%s", imsi_str);
                active_ues[ue_idx].spatial_delta = spatial_delta;
                active_ues[ue_idx].magnitude = joined_mag;
                active_ues[ue_idx].multipath_indicator = joined_csi_var;
                active_ues[ue_idx].last_updated_frame = csi_global_idx;
                active_ues[ue_idx].active = true;
            }

            // Clear old entries (more than 60000 frame increments (~30 seconds) of inactivity)
            for (int i = 0; i < MAX_ACTIVE_UES; i++) {
                if (active_ues[i].active && (csi_global_idx - active_ues[i].last_updated_frame > 60000)) {
                    active_ues[i].active = false;
                }
            }

            if (csi_global_idx % 20 == 0) {
                int active_count = 0;
                for (int i = 0; i < MAX_ACTIVE_UES; i++) {
                    if (active_ues[i].active) {
                        active_count++;
                    }
                }

                if (active_count > 0) {
                    active_ue_t sorted_ues[MAX_ACTIVE_UES];
                    memcpy(sorted_ues, active_ues, sizeof(active_ues));
                    qsort(sorted_ues, MAX_ACTIVE_UES, sizeof(active_ue_t), compare_active_ue);

                    printf("\n\033[1;37m┌─────────────────────────┬────────────────────────┬─────────────┬─────────────┐\033[0m\n");
                    printf("\033[1;37m│   TARGET IMSI (RNTI)    │       DIRECTION        │  MAGNITUDE  │  MULTIPATH  │\033[0m\n");
                    printf("\033[1;37m├─────────────────────────┼────────────────────────┼─────────────┼─────────────┤\033[0m\n");
                    char seen_imsis[MAX_ACTIVE_UES][1024];
                    int seen_count = 0;
                    for (int i = 0; i < MAX_ACTIVE_UES; i++) {
                        if (sorted_ues[i].active) {
                            bool already_seen = false;
                            for (int j = 0; j < seen_count; j++) {
                                if (strcmp(seen_imsis[j], sorted_ues[i].imsi) == 0) {
                                    already_seen = true;
                                    break;
                                }
                            }
                            if (already_seen) {
                                continue;
                            }
                            snprintf(seen_imsis[seen_count], sizeof(seen_imsis[seen_count]), "%.31s", sorted_ues[i].imsi);
                            seen_count++;

                            const char* dir_label = "CENTER";
                            const char* dir_col   = "\033[1;37m"; // White
                            if (sorted_ues[i].spatial_delta > 0.20f) { dir_label = "LEFT <<"; dir_col = "\033[1;35m"; }
                            else if (sorted_ues[i].spatial_delta < -0.20f) { dir_label = "RIGHT >>"; dir_col = "\033[1;36m"; }

                            char dir_visible[32];
                            snprintf(dir_visible, sizeof(dir_visible), "%s (%+.2f)", dir_label, sorted_ues[i].spatial_delta);
                            int dir_len = (int)strlen(dir_visible);
                            int dir_pad_left = (24 - dir_len) / 2;
                            int dir_pad_right = 24 - dir_len - dir_pad_left;

                            char ue_str[1024];
                            if (strncmp(sorted_ues[i].imsi, "RNTI-", 5) != 0) {
                                snprintf(ue_str, sizeof(ue_str), "%.31s (0x%04x)", sorted_ues[i].imsi, sorted_ues[i].rnti);
                            } else {
                                snprintf(ue_str, sizeof(ue_str), "%.31s", sorted_ues[i].imsi);
                            }
                            int ue_len = (int)strlen(ue_str);
                            int ue_pad_left = (25 - ue_len) / 2;
                            int ue_pad_right = 25 - ue_len - ue_pad_left;

                            char mag_str[32];
                            snprintf(mag_str, sizeof(mag_str), "%.2f", sorted_ues[i].magnitude);
                            int mag_len = (int)strlen(mag_str);
                            int mag_pad_left = (13 - mag_len) / 2;
                            int mag_pad_right = 13 - mag_len - mag_pad_left;

                            const char* mp_label = "LOW";
                            const char* mp_col   = "\033[1;32m"; // Green
                            if (sorted_ues[i].multipath_indicator > 0.40f) {
                                mp_label = "HIGH";
                                mp_col   = "\033[1;31m"; // Red
                            } else if (sorted_ues[i].multipath_indicator > 0.15f) {
                                mp_label = "MID";
                                mp_col   = "\033[1;33m"; // Yellow
                            }

                            char mp_visible[32];
                            snprintf(mp_visible, sizeof(mp_visible), "%s (%.2f)", mp_label, sorted_ues[i].multipath_indicator);
                            int mp_len = (int)strlen(mp_visible);
                            int mp_pad_left = (13 - mp_len) / 2;
                            int mp_pad_right = 13 - mp_len - mp_pad_left;

                            printf("\033[1;37m│\033[0m%*s%s%*s\033[1;37m│\033[0m%*s%s%s%*s\033[1;37m│\033[0m%*s%s%*s\033[1;37m│\033[0m%*s%s%s%*s\033[1;37m│\033[0m\n",
                                   ue_pad_left, "", ue_str, ue_pad_right, "",
                                   dir_pad_left, "", dir_col, dir_visible, dir_pad_right, "",
                                   mag_pad_left, "", mag_str, mag_pad_right, "",
                                   mp_pad_left, "", mp_col, mp_visible, mp_pad_right, "");
                        }
                    }
                    printf("\033[1;37m└─────────────────────────┴────────────────────────┴─────────────┴─────────────┘\033[0m\n");
                }
            }
        }
    }

    pthread_mutex_unlock(&df_state_mutex);
}

int srsran_chest_ul_estimate_pusch(srsran_chest_ul_t*     q,
                                   srsran_ul_sf_cfg_t*    sf,
                                   srsran_pusch_cfg_t*    cfg,
                                   cf_t*                  input,
                                   srsran_chest_ul_res_t* res)
{
  if (!q->dmrs_signal_configured) {
    ERROR("Error must call srsran_chest_ul_set_cfg() before using the UL estimator");
    return SRSRAN_ERROR;
  }

  uint32_t nof_prb = cfg->grant.L_prb;

  if (!srsran_dft_precoding_valid_prb(nof_prb)) {
    ERROR("Error invalid nof_prb=%d", nof_prb);
    return SRSRAN_ERROR_INVALID_INPUTS;
  }

  int nrefs_sym = nof_prb * SRSRAN_NRE;
  int nrefs_sf  = nrefs_sym * SRSRAN_NOF_SLOTS_PER_SF;

  /* Get references from the input signal */
  srsran_refsignal_dmrs_pusch_get(&q->dmrs_signal, cfg, input, q->pilot_recv_signal);

  // Use the known DMRS signal to compute Least-squares estimates
  srsran_vec_prod_conj_ccc(q->pilot_recv_signal,
                           q->dmrs_pregen.r[cfg->grant.n_dmrs][sf->tti % SRSRAN_NOF_SF_X_FRAME][nof_prb],
                           q->pilot_estimates,
                           nrefs_sf);

  // Estimate
  chest_ul_estimate(
      q, SRSRAN_NOF_SLOTS_PER_SF, nrefs_sym, 1, cfg->meas_ta_en, cfg->use_cedron_alg, true, cfg->grant.n_prb, cfg->rnti, res);

  return 0;
}

static float
estimate_noise_pilots_pucch(srsran_chest_ul_t* q, cf_t* ce, uint32_t n_rs, uint32_t n_prb[SRSRAN_NOF_SLOTS_PER_SF])
{
  float power = 0;
  for (int ns = 0; ns < SRSRAN_NOF_SLOTS_PER_SF; ns++) {
    for (int i = 0; i < n_rs; i++) {
      // All CE are the same, so pick the first symbol of the first slot always and compare with the noisy estimates
      power += srsran_chest_estimate_noise_pilots(
          &q->pilot_estimates[(i + ns * n_rs) * SRSRAN_NRE],
          &ce[SRSRAN_RE_IDX(q->cell.nof_prb, ns * SRSRAN_CP_NSYMB(q->cell.cp), n_prb[ns] * SRSRAN_NRE)],
          q->tmp_noise,
          SRSRAN_NRE);
    }
  }

  power /= (SRSRAN_NOF_SLOTS_PER_SF * n_rs);

  if (q->smooth_filter_len == 3) {
    // Calibrated for filter length 3
    float w = q->smooth_filter[0];
    float a = 7.419 * w * w + 0.1117 * w - 0.005387;
    return (power / (a * 0.8));
  } else {
    return power;
  }
}

int srsran_chest_ul_estimate_pucch(srsran_chest_ul_t*     q,
                                   srsran_ul_sf_cfg_t*    sf,
                                   srsran_pucch_cfg_t*    cfg,
                                   cf_t*                  input,
                                   srsran_chest_ul_res_t* res)
{
  int n_rs = srsran_refsignal_dmrs_N_rs(cfg->format, q->cell.cp);
  if (!n_rs) {
    ERROR("Error computing N_rs");
    return SRSRAN_ERROR;
  }
  int nrefs_sf = SRSRAN_NRE * n_rs * SRSRAN_NOF_SLOTS_PER_SF;

  /* Get references from the input signal */
  srsran_refsignal_dmrs_pucch_get(&q->dmrs_signal, cfg, input, q->pilot_recv_signal);

  /* Generate known pilots */
  if (cfg->format == SRSRAN_PUCCH_FORMAT_2A || cfg->format == SRSRAN_PUCCH_FORMAT_2B) {
    float max   = -1e9;
    int   i_max = 0;

    int m = 0;
    if (cfg->format == SRSRAN_PUCCH_FORMAT_2A) {
      m = 2;
    } else {
      m = 4;
    }

    for (int i = 0; i < m; i++) {
      cfg->pucch2_drs_bits[0] = i % 2;
      cfg->pucch2_drs_bits[1] = i / 2;
      srsran_refsignal_dmrs_pucch_gen(&q->dmrs_signal, sf, cfg, q->pilot_known_signal);
      srsran_vec_prod_conj_ccc(q->pilot_recv_signal, q->pilot_known_signal, q->pilot_estimates_tmp[i], nrefs_sf);
      float x = cabsf(srsran_vec_acc_cc(q->pilot_estimates_tmp[i], nrefs_sf));
      if (x >= max) {
        max   = x;
        i_max = i;
      }
    }
    memcpy(q->pilot_estimates, q->pilot_estimates_tmp[i_max], nrefs_sf * sizeof(cf_t));
    cfg->pucch2_drs_bits[0] = i_max % 2;
    cfg->pucch2_drs_bits[1] = i_max / 2;

  } else {
    srsran_refsignal_dmrs_pucch_gen(&q->dmrs_signal, sf, cfg, q->pilot_known_signal);
    /* Use the known DMRS signal to compute Least-squares estimates */
    srsran_vec_prod_conj_ccc(q->pilot_recv_signal, q->pilot_known_signal, q->pilot_estimates, nrefs_sf);
  }

  // Measure reference signal RE average power
  cf_t corr = srsran_vec_acc_cc(q->pilot_estimates, SRSRAN_NOF_SLOTS_PER_SF * SRSRAN_NRE * n_rs) /
              (SRSRAN_NOF_SLOTS_PER_SF * SRSRAN_NRE * n_rs);
  float rsrp_avg = __real__ corr * __real__ corr + __imag__ corr * __imag__ corr;

  // Measure EPRE
  float epre = srsran_vec_avg_power_cf(q->pilot_estimates, SRSRAN_NOF_SLOTS_PER_SF * SRSRAN_NRE * n_rs);

  // RSRP shall not be greater than EPRE
  rsrp_avg = SRSRAN_MIN(rsrp_avg, epre);

  // Set EPRE and RSRP
  res->epre      = epre;
  res->epre_dBfs = srsran_convert_power_to_dB(res->epre);
  res->rsrp      = rsrp_avg;
  res->rsrp_dBfs = srsran_convert_power_to_dB(res->rsrp);

  // Estimate time alignment
  if (cfg->meas_ta_en) {
    float ta_err = 0.0f;
    for (int ns = 0; ns < SRSRAN_NOF_SLOTS_PER_SF; ns++) {
      for (int i = 0; i < n_rs; i++) {
        if (cfg->use_cedron_alg) {
          ta_err += srsran_cedron_freq_estimate(
                        &q->srsran_cedron_freq_est, &q->pilot_estimates[(i + ns * n_rs) * SRSRAN_NRE], SRSRAN_NRE) /
                    (float)(SRSRAN_NOF_SLOTS_PER_SF * n_rs);
        } else {
          ta_err += srsran_vec_estimate_frequency(&q->pilot_estimates[(i + ns * n_rs) * SRSRAN_NRE], SRSRAN_NRE) /
                    (float)(SRSRAN_NOF_SLOTS_PER_SF * n_rs);
        }
      }
    }

    // Calculate actual time alignment error in micro-seconds
    if (isnormal(ta_err)) {
      ta_err /= 15e3f;                             // Convert from normalized frequency to seconds
      ta_err *= 1e6f;                              // Convert to micro-seconds
      ta_err     = roundf(ta_err * 10.0f) / 10.0f; // Round to one tenth of micro-second
      res->ta_us = ta_err;
    } else {
      res->ta_us = 0.0f;
    }
  }

  if (res->ce != NULL) {
    uint32_t n_prb[2] = {};

    /* TODO: Currently averaging entire slot, performance good enough? */
    for (int ns = 0; ns < 2; ns++) {
      // Average all slot
      for (int i = 1; i < n_rs; i++) {
        srsran_vec_sum_ccc(&q->pilot_estimates[ns * n_rs * SRSRAN_NRE],
                           &q->pilot_estimates[(i + ns * n_rs) * SRSRAN_NRE],
                           &q->pilot_estimates[ns * n_rs * SRSRAN_NRE],
                           SRSRAN_NRE);
      }
      srsran_vec_sc_prod_ccc(&q->pilot_estimates[ns * n_rs * SRSRAN_NRE],
                             (float)1.0 / n_rs,
                             &q->pilot_estimates[ns * n_rs * SRSRAN_NRE],
                             SRSRAN_NRE);

      // Average in freq domain
      srsran_chest_average_pilots(&q->pilot_estimates[ns * n_rs * SRSRAN_NRE],
                                  &q->pilot_recv_signal[ns * n_rs * SRSRAN_NRE],
                                  q->smooth_filter,
                                  SRSRAN_NRE,
                                  1,
                                  q->smooth_filter_len);

      // Determine n_prb
      n_prb[ns] = srsran_pucch_n_prb(&q->cell, cfg, ns);

      // copy estimates to slot
      for (int i = 0; i < SRSRAN_CP_NSYMB(q->cell.cp); i++) {
        srsran_vec_cf_copy(
            &res->ce[SRSRAN_RE_IDX(q->cell.nof_prb, i + ns * SRSRAN_CP_NSYMB(q->cell.cp), n_prb[ns] * SRSRAN_NRE)],
            &q->pilot_recv_signal[ns * n_rs * SRSRAN_NRE],
            SRSRAN_NRE);
      }
    }

    // Estimate noise/interference
    res->noise_estimate = estimate_noise_pilots_pucch(q, res->ce, n_rs, n_prb);
    if (fpclassify(res->noise_estimate) == FP_ZERO) {
      res->noise_estimate = FLT_MIN;
    }
    res->noise_estimate_dbFs = srsran_convert_power_to_dBm(res->noise_estimate);

    // Estimate SINR
    if (isnormal(res->noise_estimate)) {
      res->snr    = res->epre / res->noise_estimate;
      res->snr_db = srsran_convert_power_to_dB(res->snr);
    } else {
      res->snr    = NAN;
      res->snr_db = NAN;
    }
  }

  return 0;
}

int srsran_chest_ul_estimate_srs(srsran_chest_ul_t*                 q,
                                 srsran_ul_sf_cfg_t*                sf,
                                 srsran_refsignal_srs_cfg_t*        cfg,
                                 srsran_refsignal_dmrs_pusch_cfg_t* pusch_cfg,
                                 cf_t*                              input,
                                 srsran_chest_ul_res_t*             res)
{
  if (q == NULL || sf == NULL || cfg == NULL || pusch_cfg == NULL || input == NULL || res == NULL) {
    return SRSRAN_ERROR_INVALID_INPUTS;
  }

  // Extract parameters
  uint32_t n_srs_re = srsran_refsignal_srs_M_sc(&q->dmrs_signal, cfg);

  // Extract Sounding Reference Signal
  if (srsran_refsignal_srs_get(&q->dmrs_signal, cfg, sf->tti, q->pilot_recv_signal, input) != SRSRAN_SUCCESS) {
    return SRSRAN_ERROR;
  }

  // Get Known pilots
  cf_t* known_pilots = q->pilot_known_signal;
  if (q->srs_signal_configured) {
    known_pilots = q->srs_pregen.r[sf->tti % SRSRAN_NOF_SF_X_FRAME];
  } else {
    srsran_refsignal_srs_gen(&q->dmrs_signal, cfg, pusch_cfg, sf->tti % SRSRAN_NOF_SF_X_FRAME, known_pilots);
  }

  // Compute least squares estimates
  srsran_vec_prod_conj_ccc(q->pilot_recv_signal, known_pilots, q->pilot_estimates, n_srs_re);

  // Estimate
  uint32_t n_prb[2] = {};
  chest_ul_estimate(q, 1, n_srs_re, 1, true, false, false, n_prb, 0, res);

  return SRSRAN_SUCCESS;
}
