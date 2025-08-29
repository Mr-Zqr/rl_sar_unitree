/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "observation_buffer.hpp"

ObservationBuffer::ObservationBuffer() {}

ObservationBuffer::ObservationBuffer(int num_envs,
                                     int num_obs,
                                     int include_history_steps)
    : num_envs(num_envs),
      num_obs(num_obs),
      include_history_steps(include_history_steps)
{
    num_obs_total = num_obs * include_history_steps;
    obs_buf = torch::zeros({num_envs, num_obs_total}, torch::dtype(torch::kFloat32));
}

void ObservationBuffer::reset(std::vector<int> reset_idxs, torch::Tensor new_obs)
{
    std::vector<torch::indexing::TensorIndex> indices;
    for (int idx : reset_idxs)
    {
        indices.push_back(torch::indexing::Slice(idx));
    }
    obs_buf.index_put_(indices, new_obs.repeat({1, include_history_steps}));
}

void ObservationBuffer::insert(torch::Tensor new_obs)
{
    // Shift observations back.
    torch::Tensor shifted_obs = obs_buf.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(num_obs, num_obs * include_history_steps)}).clone();
    obs_buf.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(0, num_obs * (include_history_steps - 1))}) = shifted_obs;

    // Add new observation.
    obs_buf.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(-num_obs, torch::indexing::None)}) = new_obs;
}

/**
 * @brief Gets history of observations indexed by obs_ids.
 *
 * @param obs_ids An array of integers with which to index the desired
 *                observations, where 0 is the latest observation and
 *                include_history_steps - 1 is the oldest observation.
 * @return A torch::Tensor containing the concatenated observations.
 */
torch::Tensor ObservationBuffer::get_obs_vec(std::vector<int> obs_ids)
{
    // Define observation structure based on config: ["actions", "ang_vel", "dof_pos", "dof_vel", "gravity_vec", "g1_mimic_phase"]
    // Dimensions: actions(23), ang_vel(3), dof_pos(23), dof_vel(23), gravity_vec(3), g1_mimic_phase(1)
    const int actions_dim = 23;
    const int ang_vel_dim = 3;
    const int dof_pos_dim = 23;
    const int dof_vel_dim = 23;
    const int gravity_vec_dim = 3;
    const int g1_mimic_phase_dim = 1;
    
    // Calculate start indices for each observation type
    const int actions_start = 0;
    const int ang_vel_start = actions_start + actions_dim;
    const int dof_pos_start = ang_vel_start + ang_vel_dim;
    const int dof_vel_start = dof_pos_start + dof_pos_dim;
    const int gravity_vec_start = dof_vel_start + dof_vel_dim;
    const int g1_mimic_phase_start = gravity_vec_start + gravity_vec_dim;
    
    std::vector<torch::Tensor> obs_parts;

    std::cout << "Getting observations for IDs: ";
    
    // Find current observation (obs_id = 0) and history observations
    int current_slice_idx = -1;
    std::vector<int> history_slice_indices;
    
    for (int obs_id : obs_ids) {
        int slice_idx = include_history_steps - obs_id - 1;
        if (obs_id == 0) {
            current_slice_idx = slice_idx;
        } else {
            history_slice_indices.push_back(slice_idx);
        }
    }
    
    // Part 1: Current observation - actions, ang_vel, dof_pos, dof_vel
    if (current_slice_idx >= 0) {
        int base_idx = current_slice_idx * num_obs;
        
        // actions0
        obs_parts.push_back(obs_buf.index({
            torch::indexing::Slice(torch::indexing::None), 
            torch::indexing::Slice(base_idx + actions_start, base_idx + actions_start + actions_dim)
        }));
        
        // ang_vel0
        obs_parts.push_back(obs_buf.index({
            torch::indexing::Slice(torch::indexing::None), 
            torch::indexing::Slice(base_idx + ang_vel_start, base_idx + ang_vel_start + ang_vel_dim)
        }));
        
        // dof_pos0
        obs_parts.push_back(obs_buf.index({
            torch::indexing::Slice(torch::indexing::None), 
            torch::indexing::Slice(base_idx + dof_pos_start, base_idx + dof_pos_start + dof_pos_dim)
        }));
        
        // dof_vel0
        obs_parts.push_back(obs_buf.index({
            torch::indexing::Slice(torch::indexing::None), 
            torch::indexing::Slice(base_idx + dof_vel_start, base_idx + dof_vel_start + dof_vel_dim)
        }));
    }
    
    // Part 2: History observations grouped by feature type
    if (!history_slice_indices.empty()) {
        // Sort history indices to maintain order (1, 2, 3, 4...)
        std::sort(history_slice_indices.rbegin(), history_slice_indices.rend());
        
        // actions history (actions1, actions2, actions3, actions4)
        for (int slice_idx : history_slice_indices) {
            int base_idx = slice_idx * num_obs;
            obs_parts.push_back(obs_buf.index({
                torch::indexing::Slice(torch::indexing::None), 
                torch::indexing::Slice(base_idx + actions_start, base_idx + actions_start + actions_dim)
            }));
        }
        
        // ang_vel history (ang_vel1, ang_vel2, ang_vel3, ang_vel4)
        for (int slice_idx : history_slice_indices) {
            int base_idx = slice_idx * num_obs;
            obs_parts.push_back(obs_buf.index({
                torch::indexing::Slice(torch::indexing::None), 
                torch::indexing::Slice(base_idx + ang_vel_start, base_idx + ang_vel_start + ang_vel_dim)
            }));
        }
        
        // dof_pos history (dof_pos1, dof_pos2, dof_pos3, dof_pos4)
        for (int slice_idx : history_slice_indices) {
            int base_idx = slice_idx * num_obs;
            obs_parts.push_back(obs_buf.index({
                torch::indexing::Slice(torch::indexing::None), 
                torch::indexing::Slice(base_idx + dof_pos_start, base_idx + dof_pos_start + dof_pos_dim)
            }));
        }
        
        // dof_vel history (dof_vel1, dof_vel2, dof_vel3, dof_vel4)
        for (int slice_idx : history_slice_indices) {
            int base_idx = slice_idx * num_obs;
            obs_parts.push_back(obs_buf.index({
                torch::indexing::Slice(torch::indexing::None), 
                torch::indexing::Slice(base_idx + dof_vel_start, base_idx + dof_vel_start + dof_vel_dim)
            }));
        }
        
        // gravity_vec history (gravity_vec1, gravity_vec2, gravity_vec3, gravity_vec4)
        for (int slice_idx : history_slice_indices) {
            int base_idx = slice_idx * num_obs;
            obs_parts.push_back(obs_buf.index({
                torch::indexing::Slice(torch::indexing::None), 
                torch::indexing::Slice(base_idx + gravity_vec_start, base_idx + gravity_vec_start + gravity_vec_dim)
            }));
        }
        
        // g1_mimic_phase history (g1_mimic_phase1, g1_mimic_phase2, g1_mimic_phase3, g1_mimic_phase4)
        for (int slice_idx : history_slice_indices) {
            int base_idx = slice_idx * num_obs;
            obs_parts.push_back(obs_buf.index({
                torch::indexing::Slice(torch::indexing::None), 
                torch::indexing::Slice(base_idx + g1_mimic_phase_start, base_idx + g1_mimic_phase_start + g1_mimic_phase_dim)
            }));
        }
    }
    
    // Part 3: Current observation - gravity_vec, g1_mimic_phase
    if (current_slice_idx >= 0) {
        int base_idx = current_slice_idx * num_obs;
        
        // gravity_vec0
        obs_parts.push_back(obs_buf.index({
            torch::indexing::Slice(torch::indexing::None), 
            torch::indexing::Slice(base_idx + gravity_vec_start, base_idx + gravity_vec_start + gravity_vec_dim)
        }));
        
        // g1_mimic_phase0
        obs_parts.push_back(obs_buf.index({
            torch::indexing::Slice(torch::indexing::None), 
            torch::indexing::Slice(base_idx + g1_mimic_phase_start, base_idx + g1_mimic_phase_start + g1_mimic_phase_dim)
        }));
    }
    
    return torch::cat(obs_parts, -1);
}
