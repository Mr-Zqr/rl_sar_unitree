/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx_engine.hpp"
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <climits>
#ifdef USE_ONNXRUNTIME

ONNXInferenceEngine::ONNXInferenceEngine() 
    : env_(ORT_LOGGING_LEVEL_WARNING, "RL_SAR_ONNX"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      model_loaded_(false)
{
    session_options_.SetInterOpNumThreads(4);
    session_options_.SetIntraOpNumThreads(4);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
}

ONNXInferenceEngine::~ONNXInferenceEngine() 
{
    session_.reset();
}

void ONNXInferenceEngine::LoadModel(const std::string& model_path) 
{    
    try {
        // Check if file exists and is readable
        std::ifstream file(model_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open model file: " + model_path);
        }
        
        std::streamsize file_size = file.tellg();
        file.close();
        
        std::cout << "[ONNX Engine] Loading model: " << model_path << std::endl;
        std::cout << "[ONNX Engine] Model file size: " << file_size << " bytes" << std::endl;
        
        // Create session with additional error checking
        try {
            session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
        } catch (const Ort::Exception& e) {
            throw std::runtime_error("ONNX Runtime session creation failed: " + std::string(e.what()));
        }
        
        // Get model info
        size_t num_inputs = session_->GetInputCount();
        size_t num_outputs = session_->GetOutputCount();
        
        std::cout << "[ONNX Engine] Model has " << num_inputs << " inputs and " << num_outputs << " outputs" << std::endl;
        
        input_names_.clear();
        output_names_.clear();
        input_names_char_.clear();
        output_names_char_.clear();
        input_shapes_.clear();
        output_shapes_.clear();
        
        // Get input info with detailed validation
        for (size_t i = 0; i < num_inputs; ++i) {
            try {
                std::cout << "[ONNX Engine] Processing input " << i << "..." << std::endl;
                
                auto input_name = session_->GetInputNameAllocated(i, allocator_);
                std::cout << "[ONNX Engine] Input " << i << " name: " << input_name.get() << std::endl;
                input_names_.push_back(std::string(input_name.get()));
                input_names_char_.push_back(input_names_.back().c_str());
                
                std::cout << "[ONNX Engine] Getting input type info for input " << i << "..." << std::endl;
                auto input_type_info = session_->GetInputTypeInfo(i);
                
                std::cout << "[ONNX Engine] Getting tensor type and shape info for input " << i << "..." << std::endl;
                auto input_shape_info = input_type_info.GetTensorTypeAndShapeInfo();
                
                std::cout << "[ONNX Engine] Getting shape for input " << i << "..." << std::endl;
                auto input_shape = input_shape_info.GetShape();
                
                std::cout << "[ONNX Engine] Input " << i << " shape size: " << input_shape.size() << std::endl;
                
                // Validate input shape
                for (size_t j = 0; j < input_shape.size(); ++j) {
                    std::cout << "[ONNX Engine] Input " << i << " dimension " << j << " = " << input_shape[j] << std::endl;
                    if (input_shape[j] < -1 || input_shape[j] > 1000000) {  // Reasonable limits
                        std::cerr << "[ONNX Engine] Warning: Input " << i << " has suspicious dimension " 
                                  << j << " = " << input_shape[j] << std::endl;
                    }
                }
                
                input_shapes_.push_back(input_shape);
                std::cout << "[ONNX Engine] Input " << i << ": " << input_names_.back() << " shape: [";
                for (size_t j = 0; j < input_shape.size(); ++j) {
                    std::cout << input_shape[j];
                    if (j < input_shape.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[ONNX Engine] Exception in input processing: " << e.what() << std::endl;
                throw std::runtime_error("Error processing input " + std::to_string(i) + ": " + e.what());
            }
        }
        
        // Get output info with detailed validation
        for (size_t i = 0; i < num_outputs; ++i) {
            try {
                std::cout << "[ONNX Engine] Processing output " << i << "..." << std::endl;
                
                auto output_name = session_->GetOutputNameAllocated(i, allocator_);
                std::cout << "[ONNX Engine] Output " << i << " name: " << output_name.get() << std::endl;
                output_names_.push_back(std::string(output_name.get()));
                output_names_char_.push_back(output_names_.back().c_str());
                
                std::cout << "[ONNX Engine] Getting output type info for output " << i << "..." << std::endl;
                auto output_type_info = session_->GetOutputTypeInfo(i);
                
                std::cout << "[ONNX Engine] Getting tensor type and shape info for output " << i << "..." << std::endl;
                auto output_shape_info = output_type_info.GetTensorTypeAndShapeInfo();
                
                std::cout << "[ONNX Engine] Getting shape for output " << i << "..." << std::endl;
                auto output_shape = output_shape_info.GetShape();
                
                std::cout << "[ONNX Engine] Output " << i << " shape size: " << output_shape.size() << std::endl;
                
                // Calculate total element count with overflow protection
                size_t total_elements = 1;
                bool overflow_detected = false;
                
                // Validate output shape and check for potential overflow
                for (size_t j = 0; j < output_shape.size(); ++j) {
                    std::cout << "[ONNX Engine] Output " << i << " dimension " << j << " = " << output_shape[j] << std::endl;
                    
                    if (output_shape[j] < -1) {
                        std::cerr << "[ONNX Engine] Error: Output " << i << " has invalid negative dimension " 
                                  << j << " = " << output_shape[j] << std::endl;
                        throw std::runtime_error("Invalid negative dimension in output shape");
                    }
                    
                    if (output_shape[j] > 0) {
                        // Check for potential overflow before multiplication
                        if (total_elements > 0 && static_cast<size_t>(output_shape[j]) > SIZE_MAX / total_elements) {
                            overflow_detected = true;
                            std::cerr << "[ONNX Engine] Error: Output " << i << " dimension " << j 
                                      << " = " << output_shape[j] << " would cause overflow. "
                                      << "Total elements would exceed " << SIZE_MAX << std::endl;
                            break;
                        }
                        total_elements *= static_cast<size_t>(output_shape[j]);
                    }
                }
                
                if (overflow_detected) {
                    throw std::runtime_error("Output tensor size would exceed maximum vector size");
                }
                
                std::cout << "[ONNX Engine] Output " << i << " total elements: " << total_elements << std::endl;
                std::cout << "[ONNX Engine] Max vector size: " << std::vector<float>().max_size() << std::endl;
                
                if (total_elements > std::vector<float>().max_size()) {
                    std::cerr << "[ONNX Engine] Error: Output " << i << " requires " << total_elements 
                              << " elements, but max vector size is " << std::vector<float>().max_size() << std::endl;
                    throw std::runtime_error("Output tensor size exceeds maximum vector size");
                }
                
                output_shapes_.push_back(output_shape);
                std::cout << "[ONNX Engine] Output " << i << ": " << output_names_.back() << " shape: [";
                for (size_t j = 0; j < output_shape.size(); ++j) {
                    std::cout << output_shape[j];
                    if (j < output_shape.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[ONNX Engine] Exception in output processing: " << e.what() << std::endl;
                throw std::runtime_error("Error processing output " + std::to_string(i) + ": " + e.what());
            }
        }
        
        model_loaded_ = true;
        std::cout << "[ONNX Engine] Model loaded successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading ONNX model: " << e.what() << std::endl;
        model_loaded_ = false;
        throw;
    }
}

std::vector<float> ONNXInferenceEngine::Forward(const std::vector<float>& input_data, 
                                               const std::vector<int64_t>& input_shape) 
{
    if (!model_loaded_) {
        throw std::runtime_error("Model not loaded");
    }
    
    try {
        // Create input tensor
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, 
            const_cast<float*>(input_data.data()), 
            input_data.size(),
            input_shape.data(), 
            input_shape.size()
        );
        
        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr}, 
            input_names_char_.data(), 
            &input_tensor, 
            1,
            output_names_char_.data(), 
            output_names_char_.size()
        );
        
        // Extract output data
        if (output_tensors.empty()) {
            throw std::runtime_error("No output from model");
        }
        
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        
        std::cout << "[ONNX Engine] Forward: Output size = " << output_size << std::endl;
        std::cout << "[ONNX Engine] Forward: Max vector size = " << std::vector<float>().max_size() << std::endl;
        
        // Additional safety check before creating vector
        if (output_size > std::vector<float>().max_size()) {
            throw std::runtime_error("Output size " + std::to_string(output_size) + 
                                    " exceeds maximum vector size " + std::to_string(std::vector<float>().max_size()));
        }
        
        return std::vector<float>(output_data, output_data + output_size);
        
    } catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        throw;
    }
}

void ONNXInferenceEngine::PrintModelInfo() 
{
    std::cout << "[ONNX Engine] Model loaded successfully" << std::endl;
    std::cout << "[ONNX Engine] Inputs: " << input_names_.size() << std::endl;
    for (size_t i = 0; i < input_names_.size(); ++i) {
        std::cout << "  Input " << i << ": " << input_names_[i] << " [";
        for (size_t j = 0; j < input_shapes_[i].size(); ++j) {
            std::cout << input_shapes_[i][j];
            if (j < input_shapes_[i].size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    std::cout << "[ONNX Engine] Outputs: " << output_names_.size() << std::endl;
    for (size_t i = 0; i < output_names_.size(); ++i) {
        std::cout << "  Output " << i << ": " << output_names_[i] << " [";
        for (size_t j = 0; j < output_shapes_[i].size(); ++j) {
            std::cout << output_shapes_[i][j];
            if (j < output_shapes_[i].size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

#else // USE_ONNXRUNTIME not defined

// Dummy implementation when ONNX Runtime is not available
ONNXInferenceEngine::ONNXInferenceEngine() : model_loaded_(false)
{
    std::cout << "[ONNX Engine] ONNX Runtime not available, using PyTorch only" << std::endl;
}

ONNXInferenceEngine::~ONNXInferenceEngine() 
{
}

void ONNXInferenceEngine::LoadModel(const std::string& model_path) 
{
    std::cout << "[ONNX Engine] ONNX Runtime not available, cannot load " << model_path << std::endl;
    model_loaded_ = false;
}

std::vector<float> ONNXInferenceEngine::Forward(const std::vector<float>& input_data, 
                                               const std::vector<int64_t>& input_shape) 
{
    throw std::runtime_error("ONNX Runtime not available");
}

#endif // USE_ONNXRUNTIME
