/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_logger.hpp"
#include <filesystem>

RLLogger::RLLogger()
{
    InitJointNames();
}

RLLogger::~RLLogger()
{
    // 析构时不自动保存，需要显式调用SaveToCSV
}

void RLLogger::InitJointNames()
{
    joint_names_[0] = "L_hip_pitch";
    joint_names_[1] = "L_hip_roll";
    joint_names_[2] = "L_hip_yaw";
    joint_names_[3] = "L_knee";
    joint_names_[4] = "L_ankle_pitch";
    joint_names_[5] = "L_ankle_roll";
    joint_names_[6] = "R_hip_pitch";
    joint_names_[7] = "R_hip_roll";
    joint_names_[8] = "R_hip_yaw";
    joint_names_[9] = "R_knee";
    joint_names_[10] = "R_ankle_pitch";
    joint_names_[11] = "R_ankle_roll";
    joint_names_[12] = "Waist_yaw";
    joint_names_[13] = "Waist_roll";
    joint_names_[14] = "Waist_pitch";
    joint_names_[15] = "L_shoulder_pitch";
    joint_names_[16] = "L_shoulder_roll";
    joint_names_[17] = "L_shoulder_yaw";
    joint_names_[18] = "L_elbow";
    joint_names_[19] = "L_wrist_roll";
    joint_names_[20] = "L_wrist_pitch";
    joint_names_[21] = "L_wrist_yaw";
    joint_names_[22] = "R_shoulder_pitch";
    joint_names_[23] = "R_shoulder_roll";
    joint_names_[24] = "R_shoulder_yaw";
    joint_names_[25] = "R_elbow";
    joint_names_[26] = "R_wrist_roll";
    joint_names_[27] = "R_wrist_pitch";
    joint_names_[28] = "R_wrist_yaw";
}

std::string RLLogger::GetJointName(int joint_index) const
{
    auto it = joint_names_.find(joint_index);
    if (it != joint_names_.end()) {
        return it->second;
    }
    return "joint_" + std::to_string(joint_index);
}

void RLLogger::Record(const std::string& key, double value)
{
    data_[key].push_back(value);
}

void RLLogger::RecordJointData(int joint_index, double target_q, double actual_q, 
                              double actual_dq, double kp, double kd, double tau_est)
{
    std::string joint_name = GetJointName(joint_index);
    
    // 记录关节数据，按照deploy_logger的命名格式
    Record(joint_name + "_target", target_q);
    Record(joint_name + "_actual", actual_q);
    Record(joint_name + "_dq", actual_dq);
    Record(joint_name + "_kp", kp);
    Record(joint_name + "_kd", kd);
    Record(joint_name + "_tau_est", tau_est);
}

std::string RLLogger::GenerateFilename() const
{
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << "log/robot_control_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".csv";
    return ss.str();
}

void RLLogger::SaveToCSV(const std::string& filename)
{
    if (data_.empty()) {
        std::cout << "⚠️  No data to save" << std::endl;
        return;
    }
    
    // 创建log目录（如果不存在）
    std::filesystem::create_directories("log");
    
    // 生成文件名（如果未提供）
    std::string output_filename = filename.empty() ? GenerateFilename() : filename;
    
    // 获取所有键（列名）
    std::vector<std::string> columns;
    for (const auto& pair : data_) {
        columns.push_back(pair.first);
    }
    
    // 获取最大行数
    size_t max_rows = 0;
    for (const auto& pair : data_) {
        max_rows = std::max(max_rows, pair.second.size());
    }
    
    // 写入CSV文件
    std::ofstream file(output_filename);
    if (!file.is_open()) {
        std::cout << "❌ Error opening file: " << output_filename << std::endl;
        return;
    }
    
    // 写入列名
    for (size_t i = 0; i < columns.size(); ++i) {
        file << columns[i];
        if (i < columns.size() - 1) file << ",";
    }
    file << "\n";
    
    // 逐行写入数据
    for (size_t row = 0; row < max_rows; ++row) {
        for (size_t col = 0; col < columns.size(); ++col) {
            const std::string& key = columns[col];
            const auto& values = data_.at(key);
            
            if (row < values.size()) {
                file << values[row];
            }
            // 如果该列的数据已经用完，留空
            
            if (col < columns.size() - 1) file << ",";
        }
        file << "\n";
    }
    
    file.close();
    
    std::cout << "📊 Data saved to: " << output_filename << std::endl;
    std::cout << "📈 Total records: " << max_rows << std::endl;
    std::cout << "📋 Columns: " << columns.size() << std::endl;
}

void RLLogger::Clear()
{
    data_.clear();
}

std::string RLLogger::GetSummary() const
{
    if (data_.empty()) {
        return "No data recorded";
    }
    
    size_t total_records = 0;
    for (const auto& pair : data_) {
        total_records = std::max(total_records, pair.second.size());
    }
    
    return "Records: " + std::to_string(total_records) + 
           ", Columns: " + std::to_string(data_.size());
}

bool RLLogger::HasData() const
{
    return !data_.empty();
}
