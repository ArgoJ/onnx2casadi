"""
ONNX importer module for loading and parsing ONNX models.
"""
from typing import Dict, Any, List
import os
import onnx
from onnx import ModelProto, ValueInfoProto


class OnnxImporter:
    """
    Handles loading and parsing ONNX models.
    
    This class is responsible for reading ONNX model files and extracting
    relevant information such as input/output specifications and the graph structure.
    """
    
    def __init__(self):
        """Initialize the ONNX importer."""
        pass
        
    def load_model(self, model_path: str) -> ModelProto:
        """
        Load an ONNX model from the specified file path.
        
        Args:
            model_path (str): Path to the ONNX model file
            
        Returns:
            ModelProto: The loaded ONNX model
            
        Raises:
            FileNotFoundError: If the model file doesn't exist
            ValueError: If the model file is invalid or cannot be parsed
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        try:
            model = onnx.load(model_path)
            # Validate the model
            onnx.checker.check_model(model)
            return model
        except Exception as e:
            raise ValueError(f"Failed to load ONNX model: {str(e)}")
            
    def get_inputs(self, model: ModelProto) -> Dict[str, Any]:
        """
        Extract input tensor information from the ONNX model.
        
        Args:
            model (ModelProto): The ONNX model
            
        Returns:
            Dict[str, Any]: Dictionary mapping input names to their properties
                           (shape, dtype, etc.)
        """
        inputs = {}
        for input_tensor in model.graph.input:
            inputs[input_tensor.name] = self._parse_value_info(input_tensor)
        return inputs
        
    def get_outputs(self, model: ModelProto) -> Dict[str, Any]:
        """
        Extract output tensor information from the ONNX model.
        
        Args:
            model (ModelProto): The ONNX model
            
        Returns:
            Dict[str, Any]: Dictionary mapping output names to their properties
                           (shape, dtype, etc.)
        """
        outputs = {}
        for output_tensor in model.graph.output:
            outputs[output_tensor.name] = self._parse_value_info(output_tensor)
        return outputs
        
    def _parse_value_info(self, value_info: ValueInfoProto) -> Dict[str, Any]:
        """
        Parse a ValueInfoProto to extract tensor information.
        
        Args:
            value_info (ValueInfoProto): ONNX value info object
            
        Returns:
            Dict[str, Any]: Dictionary containing tensor properties
        """
        info = {
            'name': value_info.name,
        }
        
        if value_info.type.HasField('tensor_type'):
            tensor_type = value_info.type.tensor_type
            
            # Extract data type
            if tensor_type.HasField('elem_type'):
                info['dtype'] = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
                
            # Extract shape
            if tensor_type.HasField('shape'):
                shape = []
                for dim in tensor_type.shape.dim:
                    if dim.HasField('dim_value'):
                        shape.append(dim.dim_value)
                    elif dim.HasField('dim_param'):
                        shape.append(dim.dim_param)
                    else:
                        shape.append(None)
                info['shape'] = shape
                
        return info
