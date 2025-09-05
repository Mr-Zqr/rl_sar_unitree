/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ONNX_ENGINE_HPP
#define ONNX_ENGINE_HPP

// #ifdef USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
// #include "/home/unitree/lib/libonnxruntime/include/onnxruntime_cxx_api.h"
// #endif

#include <vector>
#include <memory>
#include <string>
#include <iostream>

class ONNXInferenceEngine 
{
public:
    ONNXInferenceEngine();
    ~ONNXInferenceEngine();
    
    void LoadModel(const std::string& model_path);
// #ifdef USE_ONNXRUNTIME
    std::vector<Ort::Value> Forward(const std::vector<float>& obs, 
                                               const float & time_step);

// #else
    std::vector<Ort::Value> FirstOutput();
// #endif
    
    bool IsModelLoaded() const { return model_loaded_; }
    
// #ifdef USE_ONNXRUNTIME
    // Helper methods for working with output tensors
    static std::vector<float> ExtractTensorData(const Ort::Value& tensor);
    static std::vector<int64_t> GetTensorShape(const Ort::Value& tensor);
    static size_t GetTensorElementCount(const Ort::Value& tensor);
    static ONNXTensorElementDataType GetTensorDataType(const Ort::Value& tensor);
    static std::string GetTensorDataTypeString(const Ort::Value& tensor);
    
    // Get output names for indexing the results
    const std::vector<std::string>& GetOutputNames() const { return output_names_; }
    const std::vector<std::string>& GetInputNames() const { return input_names_; }
    bool model_loaded_;
// #endif
    
private:
// #ifdef USE_ONNXRUNTIME
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::SessionOptions session_options_;
    Ort::MemoryInfo memory_info_;
    Ort::AllocatorWithDefaultOptions allocator_;
    
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_names_char_;
    std::vector<const char*> output_names_char_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
    
    void PrintModelInfo();
#endif
    
};

// #endif // ONNX_ENGINE_HPP
