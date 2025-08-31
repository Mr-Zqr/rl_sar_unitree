/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx_engine.hpp"
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <climits>
#include <cstring>

// #ifdef USE_ONNXRUNTIME

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
                std::string input_name_str = std::string(input_name.get());
                std::cout << "[ONNX Engine] Input " << i << " name: '" << input_name_str << "'" << std::endl;
                
                // Check if input name is empty and generate a default name if needed
                if (input_name_str.empty()) {
                    input_name_str = "input_" + std::to_string(i);
                    std::cout << "[ONNX Engine] Warning: Input " << i << " has empty name, using default: '" << input_name_str << "'" << std::endl;
                }
                
                input_names_.push_back(input_name_str);
                // Don't convert to c_str() yet - will do after all names are collected
                
                std::cout << "[ONNX Engine] Getting input type info for input " << i << "..." << std::endl;
                auto input_type_info = session_->GetInputTypeInfo(i);
                
                std::cout << "[ONNX Engine] Getting tensor type and shape info for input " << i << "..." << std::endl;
                auto input_shape_info = input_type_info.GetTensorTypeAndShapeInfo();
                
                std::cout << "[ONNX Engine] Getting shape for input " << i << "..." << std::endl;
                auto input_shape = input_shape_info.GetShape();
                
                std::cout << "[ONNX Engine] Input " << i << " shape size: " << input_shape.size() << std::endl;
                
                // Get and print input data type information
                ONNXTensorElementDataType input_data_type = input_shape_info.GetElementType();
                std::string input_type_name;
                switch (input_data_type) {
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                        input_type_name = "FLOAT32"; break;
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                        input_type_name = "DOUBLE"; break;
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                        input_type_name = "INT32"; break;
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                        input_type_name = "INT64"; break;
                    default:
                        input_type_name = "OTHER(" + std::to_string(static_cast<int>(input_data_type)) + ")"; break;
                }
                std::cout << "[ONNX Engine] Input " << i << " data type: " << input_type_name << std::endl;
                
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
                std::string output_name_str = std::string(output_name.get());
                std::cout << "[ONNX Engine] Output " << i << " name: '" << output_name_str << "'" << std::endl;
                
                // Check if output name is empty and generate a default name if needed
                if (output_name_str.empty()) {
                    output_name_str = "output_" + std::to_string(i);
                    std::cout << "[ONNX Engine] Warning: Output " << i << " has empty name, using default: '" << output_name_str << "'" << std::endl;
                }
                
                output_names_.push_back(output_name_str);
                // Don't convert to c_str() yet - will do after all names are collected
                
                std::cout << "[ONNX Engine] Getting output type info for output " << i << "..." << std::endl;
                auto output_type_info = session_->GetOutputTypeInfo(i);
                
                std::cout << "[ONNX Engine] Getting tensor type and shape info for output " << i << "..." << std::endl;
                auto output_shape_info = output_type_info.GetTensorTypeAndShapeInfo();
                
                std::cout << "[ONNX Engine] Getting shape for output " << i << "..." << std::endl;
                auto output_shape = output_shape_info.GetShape();
                
                std::cout << "[ONNX Engine] Output " << i << " shape size: " << output_shape.size() << std::endl;
                
                // Get and print data type information
                ONNXTensorElementDataType data_type = output_shape_info.GetElementType();
                std::string type_name;
                switch (data_type) {
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
                        type_name = "UNDEFINED"; break;
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                        type_name = "FLOAT32"; break;
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                        type_name = "UINT8"; break;
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                        type_name = "INT8"; break;
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
                        type_name = "UINT16"; break;
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                        type_name = "INT16"; break;
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                        type_name = "INT32"; break;
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                        type_name = "INT64"; break;
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
                        type_name = "STRING"; break;
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
                        type_name = "BOOL"; break;
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
                        type_name = "FLOAT16"; break;
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                        type_name = "DOUBLE"; break;
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
                        type_name = "UINT32"; break;
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
                        type_name = "UINT64"; break;
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
                        type_name = "COMPLEX64"; break;
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
                        type_name = "COMPLEX128"; break;
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
                        type_name = "BFLOAT16"; break;
                    default:
                        type_name = "UNKNOWN(" + std::to_string(static_cast<int>(data_type)) + ")"; break;
                }
                std::cout << "[ONNX Engine] Output " << i << " data type: " << type_name << std::endl;
                
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
        
        // Now that all names are collected, convert to const char* pointers
        input_names_char_.clear();
        output_names_char_.clear();
        
        for (const auto& name : input_names_) {
            input_names_char_.push_back(name.c_str());
        }
        
        for (const auto& name : output_names_) {
            output_names_char_.push_back(name.c_str());
        }
        
        // Validate that we have the expected number of names
        std::cout << "[ONNX Engine] Final validation: " << input_names_char_.size() 
                  << " input names, " << output_names_char_.size() << " output names" << std::endl;
        
        for (size_t i = 0; i < input_names_char_.size(); ++i) {
            if (input_names_char_[i] == nullptr || strlen(input_names_char_[i]) == 0) {
                throw std::runtime_error("Input name " + std::to_string(i) + " is null or empty");
            }
            std::cout << "[ONNX Engine] Input name " << i << ": '" << input_names_char_[i] << "'" << std::endl;
        }
        
        for (size_t i = 0; i < output_names_char_.size(); ++i) {
            if (output_names_char_[i] == nullptr || strlen(output_names_char_[i]) == 0) {
                throw std::runtime_error("Output name " + std::to_string(i) + " is null or empty");
            }
            std::cout << "[ONNX Engine] Output name " << i << ": '" << output_names_char_[i] << "'" << std::endl;
        }
        
        model_loaded_ = true;
        std::cout << "[ONNX Engine] Model loaded successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading ONNX model: " << e.what() << std::endl;
        model_loaded_ = false;
        throw;
    }
}

std::vector<Ort::Value> ONNXInferenceEngine::FirstOutput() 
{
    if (!model_loaded_) {
        throw std::runtime_error("Model not loaded");
    }
    
    try {
        // Create dummy inputs with all zeros
        std::vector<Ort::Value> input_tensors;
        std::vector<std::vector<float>> dummy_inputs; // Keep data alive
        
        for (size_t i = 0; i < input_shapes_.size(); ++i) {
            // Calculate the total number of elements for this input
            size_t total_elements = 1;
            for (int64_t dim : input_shapes_[i]) {
                if (dim > 0) {
                    total_elements *= static_cast<size_t>(dim);
                }
            }
            
            // Create dummy input data filled with zeros
            dummy_inputs.emplace_back(total_elements, 0.0f);
            
            // Create input tensor
            auto input_tensor = Ort::Value::CreateTensor<float>(
                memory_info_, 
                dummy_inputs.back().data(), 
                dummy_inputs.back().size(),
                input_shapes_[i].data(), 
                input_shapes_[i].size()
            );
            
            input_tensors.push_back(std::move(input_tensor));
        }
        
        // Run inference with all dummy inputs
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr}, 
            input_names_char_.data(), 
            input_tensors.data(), 
            input_tensors.size(),
            output_names_char_.data(), 
            output_names_char_.size()
        );
        
        if (output_tensors.empty()) {
            throw std::runtime_error("No output from model");
        }
        
        return output_tensors;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during FirstOutput inference: " << e.what() << std::endl;
        throw;
    }
}

std::vector<Ort::Value> ONNXInferenceEngine::Forward(const std::vector<float>& obs, 
                                               const float & time_step) 
{
    if (!model_loaded_) {
        throw std::runtime_error("Model not loaded");
    }
    
    try {
        // Create input tensor
        std::vector<Ort::Value> input_tensors;


        // Keep data alive during inference - similar to FirstOutput approach
        std::vector<float> time_step_data = { time_step};

        auto input_tensor_obs = Ort::Value::CreateTensor<float>(
            memory_info_,
            const_cast<float*>(obs.data()),
            obs.size(),
            input_shapes_[0].data(),
            input_shapes_[0].size()
        );

        auto input_tensor_time_step = Ort::Value::CreateTensor<float>(
            memory_info_,
            time_step_data.data(),
            time_step_data.size(),
            input_shapes_[1].data(),
            input_shapes_[1].size()
        );

        input_tensors.push_back(std::move(input_tensor_obs));
        input_tensors.push_back(std::move(input_tensor_time_step));

        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr}, 
            input_names_char_.data(), 
            input_tensors.data(), 
            input_tensors.size(),
            output_names_char_.data(), 
            output_names_char_.size()
        );
        
        // Extract output data
        if (output_tensors.empty()) {
            throw std::runtime_error("No output from model");
        }
        
        return output_tensors;
        
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

// Helper methods for working with output tensors
std::vector<float> ONNXInferenceEngine::ExtractTensorData(const Ort::Value& tensor) 
{
    float* data = const_cast<Ort::Value&>(tensor).GetTensorMutableData<float>();
    size_t size = tensor.GetTensorTypeAndShapeInfo().GetElementCount();
    return std::vector<float>(data, data + size);
}

std::vector<int64_t> ONNXInferenceEngine::GetTensorShape(const Ort::Value& tensor) 
{
    return tensor.GetTensorTypeAndShapeInfo().GetShape();
}

size_t ONNXInferenceEngine::GetTensorElementCount(const Ort::Value& tensor) 
{
    return tensor.GetTensorTypeAndShapeInfo().GetElementCount();
}

ONNXTensorElementDataType ONNXInferenceEngine::GetTensorDataType(const Ort::Value& tensor) 
{
    return tensor.GetTensorTypeAndShapeInfo().GetElementType();
}

std::string ONNXInferenceEngine::GetTensorDataTypeString(const Ort::Value& tensor) 
{
    ONNXTensorElementDataType type = tensor.GetTensorTypeAndShapeInfo().GetElementType();
   
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            return "UNDEFINED";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return "FLOAT32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return "UINT8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return "INT8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            return "UINT16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            return "INT16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return "INT32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return "INT64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            return "STRING";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return "BOOL";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return "FLOAT16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return "DOUBLE";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            return "UINT32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            return "UINT64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            return "COMPLEX64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            return "COMPLEX128";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            return "BFLOAT16";
        default:
            return "UNKNOWN(" + std::to_string(static_cast<int>(type)) + ")";
    }
}

// #else // USE_ONNXRUNTIME not defined

// Dummy implementation when ONNX Runtime is not available
// ONNXInferenceEngine::ONNXInferenceEngine() : model_loaded_(false)
// {
//     std::cout << "[ONNX Engine] ONNX Runtime not available, using PyTorch only" << std::endl;
// }

// ONNXInferenceEngine::~ONNXInferenceEngine() 
// {
// }

// void ONNXInferenceEngine::LoadModel(const std::string& model_path) 
// {
//     std::cout << "[ONNX Engine] ONNX Runtime not available, cannot load " << model_path << std::endl;
//     model_loaded_ = false;
// }

// std::vector<float> ONNXInferenceEngine::FirstOutput() 
// {
//     throw std::runtime_error("ONNX Runtime not available");
// }

// std::vector<float> ONNXInferenceEngine::Forward(const std::vector<float>& input_data, 
//                                                const std::vector<int64_t>& input_shape) 
// {
//     throw std::runtime_error("ONNX Runtime not available");
// }

// #endif // USE_ONNXRUNTIME
