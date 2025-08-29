// Example usage of the improved FirstOutput() method
#ifdef USE_ONNXRUNTIME
#include "src/rl_sar/library/core/onnx_engine/onnx_engine.hpp"

void example_usage() {
    ONNXInferenceEngine engine;
    
    // Load your model
    engine.LoadModel("path/to/your/model.onnx");
    
    // Get all outputs with dummy zero inputs
    auto outputs = engine.FirstOutput();
    
    // Access output names
    auto output_names = engine.GetOutputNames();
    
    // Process each output tensor
    for (size_t i = 0; i < outputs.size(); ++i) {
        std::cout << "Output " << i << " (" << output_names[i] << "):" << std::endl;
        
        // Get tensor data type
        std::string data_type = ONNXInferenceEngine::GetTensorDataTypeString(outputs[i]);
        std::cout << "  Data type: " << data_type << std::endl;
        
        // Get tensor shape
        auto shape = ONNXInferenceEngine::GetTensorShape(outputs[i]);
        std::cout << "  Shape: [";
        for (size_t j = 0; j < shape.size(); ++j) {
            std::cout << shape[j];
            if (j < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Get element count
        size_t element_count = ONNXInferenceEngine::GetTensorElementCount(outputs[i]);
        std::cout << "  Element count: " << element_count << std::endl;
        
        // Extract data based on type (this example assumes float)
        if (data_type == "FLOAT32") {
            auto data = ONNXInferenceEngine::ExtractTensorData(outputs[i]);
            std::cout << "  First few values: ";
            for (size_t j = 0; j < std::min(size_t(5), data.size()); ++j) {
                std::cout << data[j] << " ";
            }
            std::cout << std::endl;
            
            // Or access data directly without copying
            float* raw_data = outputs[i].GetTensorMutableData<float>();
            // Use raw_data directly...
        } else {
            std::cout << "  Non-float data type - handle accordingly" << std::endl;
        }
    }
}
#endif
