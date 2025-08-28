#!/usr/bin/env python3
"""
Script to convert PyTorch models to ONNX format for RL_SAR project.

Usage:
    python convert_to_onnx.py <model_path> [--input_size INPUT_SIZE] [--output_path OUTPUT_PATH]

Examples:
    python convert_to_onnx.py policy/g1/policy.pt
    python convert_to_onnx.py policy/go2/policy.pt --input_size 48
"""

import torch
import torch.onnx
import argparse
import os
import sys

def convert_pytorch_to_onnx(model_path, input_size=None, output_path=None):
    """
    Convert PyTorch JIT model to ONNX format.
    
    Args:
        model_path (str): Path to PyTorch .pt model file
        input_size (int): Input observation size (if None, will try to infer)
        output_path (str): Output ONNX file path (if None, will replace .pt with .onnx)
    """
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return False
    
    if output_path is None:
        output_path = model_path.replace('.pt', '.onnx')
    
    try:
        # Load the PyTorch model
        print(f"Loading PyTorch model: {model_path}")
        model = torch.jit.load(model_path)
        model.eval()
        
        # If input size is not provided, try common sizes
        if input_size is None:
            common_sizes = [45, 48, 51, 57, 87]  # Common observation sizes for different robots
            input_size = 48  # Default for most robots
            print(f"Warning: Input size not specified, using default: {input_size}")
            print(f"If this fails, try specifying --input_size with one of: {common_sizes}")
        
        # Create dummy input
        dummy_input = torch.randn(1, input_size, dtype=torch.float32)
        
        # Test the model with dummy input
        with torch.no_grad():
            output = model(dummy_input)
            print(f"Model test successful - Input: {dummy_input.shape}, Output: {output.shape}")
        
        # Export to ONNX
        print(f"Exporting to ONNX: {output_path}")
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['observations'],
            output_names=['actions'],
            dynamic_axes={
                'observations': {0: 'batch_size'},
                'actions': {0: 'batch_size'}
            }
        )
        
        print(f"Successfully converted {model_path} to {output_path}")
        
        # Verify the ONNX model
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model verification successful")
        except ImportError:
            print("Warning: onnx package not available for verification")
            print("You can install it with: pip install onnx")
        except Exception as e:
            print(f"Warning: ONNX model verification failed: {e}")
            print("Model was still exported successfully")
        
        return True
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch models to ONNX format')
    parser.add_argument('model_path', help='Path to PyTorch .pt model file')
    parser.add_argument('--input_size', type=int, help='Input observation size')
    parser.add_argument('--output_path', help='Output ONNX file path')
    parser.add_argument('--batch_convert', action='store_true', 
                       help='Convert all .pt files in the policy directory')
    
    args = parser.parse_args()
    
    if args.batch_convert:
        # Convert all .pt files in policy directory
        policy_dir = os.path.join(os.path.dirname(__file__), 'policy')
        if not os.path.exists(policy_dir):
            print(f"Error: Policy directory {policy_dir} not found")
            return
        
        success_count = 0
        total_count = 0
        
        for root, dirs, files in os.walk(policy_dir):
            for file in files:
                if file.endswith('.pt'):
                    model_path = os.path.join(root, file)
                    print(f"\n--- Converting {model_path} ---")
                    total_count += 1
                    if convert_pytorch_to_onnx(model_path, args.input_size):
                        success_count += 1
        
        print(f"\nBatch conversion completed: {success_count}/{total_count} models converted successfully")
    else:
        # Convert single model
        success = convert_pytorch_to_onnx(args.model_path, args.input_size, args.output_path)
        sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
