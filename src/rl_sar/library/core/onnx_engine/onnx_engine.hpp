/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ONNX_ENGINE_HPP
#define ONNX_ENGINE_HPP

#ifdef USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

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
    std::vector<float> Forward(const std::vector<float>& input_data, 
                              const std::vector<int64_t>& input_shape);
    
    bool IsModelLoaded() const { return model_loaded_; }
    
private:
#ifdef USE_ONNXRUNTIME
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
    
    bool model_loaded_;
};

#endif // ONNX_ENGINE_HPP
