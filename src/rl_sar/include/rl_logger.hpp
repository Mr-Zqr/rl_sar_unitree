/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RL_LOGGER_HPP
#define RL_LOGGER_HPP

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

class RLLogger
{
public:
    RLLogger();
    ~RLLogger();

    // 记录数据的通用接口
    void Record(const std::string& key, double value);
    
    // 记录关节数据的便捷接口
    void RecordJointData(int joint_index, double target_q, double actual_q, 
                        double actual_dq, double kp, double kd, double tau_est);
    
    // 保存数据到CSV文件
    void SaveToCSV(const std::string& filename = "");
    
    // 清空所有记录的数据
    void Clear();
    
    // 获取数据摘要
    std::string GetSummary() const;
    
    // 检查是否有数据
    bool HasData() const;

private:
    std::map<std::string, std::vector<double>> data_;
    
    // 关节名称映射
    std::map<int, std::string> joint_names_;
    
    // 初始化关节名称映射
    void InitJointNames();
    
    // 获取关节名称
    std::string GetJointName(int joint_index) const;
    
    // 生成时间戳文件名
    std::string GenerateFilename() const;
};

#endif // RL_LOGGER_HPP
