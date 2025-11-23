"""
Main class for converting ONNX models to CasADi symbolic graphs.
"""
from typing import Dict, Optional, Any
import onnx
from onnx import ModelProto
import casadi as ca

from .importers.onnx_importer import OnnxImporter
from .layers.layer_converter import LayerConverter


class OnnxToCasadi:
    """
    Main class for converting ONNX models to CasADi symbolic graphs.
    
    This class loads an ONNX model and converts it into a CasADi symbolic
    representation (using casadi.MX) that can be used for optimization and
    control applications.
    
    Attributes:
        model_path (str): Path to the ONNX model file
        model (ModelProto): The loaded ONNX model
        casadi_graph (ca.MX): The resulting CasADi symbolic graph
        importer (OnnxImporter): ONNX model importer
        converter (LayerConverter): Layer-to-CasADi converter
        
    Example:
        >>> converter = OnnxToCasadi("model.onnx")
        >>> converter.load()
        >>> mx_graph = converter.convert()
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the ONNX to CasADi converter.
        
        Args:
            model_path (str): Path to the ONNX model file
        """
        self.model_path = model_path
        self.model: Optional[ModelProto] = None
        self.casadi_graph: Optional[ca.MX] = None
        self.importer = OnnxImporter()
        self.converter = LayerConverter()
        
    def load(self) -> 'OnnxToCasadi':
        """
        Load the ONNX model from the specified path.
        
        Returns:
            OnnxToCasadi: Self for method chaining
            
        Raises:
            FileNotFoundError: If the model file doesn't exist
            ValueError: If the model file is invalid
        """
        self.model = self.importer.load_model(self.model_path)
        return self
        
    def convert(self) -> ca.MX:
        """
        Convert the loaded ONNX model to a CasADi symbolic graph.
        
        Returns:
            ca.MX: The CasADi symbolic graph
            
        Raises:
            RuntimeError: If the model hasn't been loaded yet
            NotImplementedError: If the model contains unsupported operations
        """
        if self.model is None:
            raise RuntimeError("Model must be loaded before conversion. Call load() first.")
            
        self.casadi_graph = self.converter.convert_model(self.model)
        return self.casadi_graph
        
    def get_inputs(self) -> Dict[str, Any]:
        """
        Get information about the model's input tensors.
        
        Returns:
            Dict[str, Any]: Dictionary mapping input names to their properties
            
        Raises:
            RuntimeError: If the model hasn't been loaded yet
        """
        if self.model is None:
            raise RuntimeError("Model must be loaded before accessing inputs. Call load() first.")
            
        return self.importer.get_inputs(self.model)
        
    def get_outputs(self) -> Dict[str, Any]:
        """
        Get information about the model's output tensors.
        
        Returns:
            Dict[str, Any]: Dictionary mapping output names to their properties
            
        Raises:
            RuntimeError: If the model hasn't been loaded yet
        """
        if self.model is None:
            raise RuntimeError("Model must be loaded before accessing outputs. Call load() first.")
            
        return self.importer.get_outputs(self.model)
