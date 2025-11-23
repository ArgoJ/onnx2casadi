"""
Basic usage example for onnx2casadi.

This example demonstrates how to use the OnnxToCasadi converter
to load an ONNX model and convert it to CasADi symbolic format.
"""
from onnx2casadi import OnnxToCasadi


def main():
    """
    Main example function.
    
    Note: This is a template. You'll need an actual ONNX model file
    to run this example successfully.
    """
    # Path to your ONNX model
    model_path = "model.onnx"
    
    # Create converter instance
    print(f"Loading model from: {model_path}")
    converter = OnnxToCasadi(model_path)
    
    # Load the ONNX model
    try:
        converter.load()
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print(f"✗ Error: Model file '{model_path}' not found")
        print("  Please provide a valid ONNX model file to run this example.")
        return
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Get model information
    inputs = converter.get_inputs()
    outputs = converter.get_outputs()
    
    print(f"\nModel Inputs:")
    for name, info in inputs.items():
        print(f"  - {name}: shape={info.get('shape')}, dtype={info.get('dtype')}")
    
    print(f"\nModel Outputs:")
    for name, info in outputs.items():
        print(f"  - {name}: shape={info.get('shape')}, dtype={info.get('dtype')}")
    
    # Convert to CasADi
    try:
        casadi_graph = converter.convert()
        print(f"\n✓ Model converted to CasADi successfully")
        print(f"  CasADi graph type: {type(casadi_graph)}")
        print(f"  Graph dimensions: {casadi_graph.shape}")
    except NotImplementedError as e:
        print(f"\n✗ Conversion error: {e}")
        print("  The model contains operations that are not yet supported.")
    except Exception as e:
        print(f"\n✗ Error during conversion: {e}")


if __name__ == "__main__":
    main()
