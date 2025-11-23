"""
Layer converter module for converting ONNX operations to CasADi symbolic operations.
"""
from typing import Dict, Any, List
import casadi as ca
from onnx import ModelProto, NodeProto
import numpy as np


class LayerConverter:
    """
    Converts ONNX operations to CasADi symbolic operations.
    
    This class handles the conversion of ONNX graph nodes (operations) into
    their equivalent CasADi symbolic representations using casadi.MX.
    """
    
    def __init__(self):
        """Initialize the layer converter."""
        self.supported_ops = {
            'Add', 'Sub', 'Mul', 'Div', 'MatMul', 'Gemm',
            'Relu', 'Sigmoid', 'Tanh', 'Identity',
            'Constant', 'Reshape', 'Transpose', 'Concat'
        }
        
    def convert_model(self, model: ModelProto) -> ca.MX:
        """
        Convert an ONNX model to a CasADi symbolic graph.
        
        Args:
            model (ModelProto): The ONNX model to convert
            
        Returns:
            ca.MX: The resulting CasADi symbolic graph
            
        Raises:
            NotImplementedError: If the model contains unsupported operations
        """
        # Create a dictionary to store intermediate results
        values: Dict[str, ca.MX] = {}
        
        # Create symbolic inputs
        for input_tensor in model.graph.input:
            shape = self._get_tensor_shape(input_tensor)
            if shape:
                # Create a symbolic variable with the appropriate shape
                total_size = int(np.prod([d for d in shape if isinstance(d, int)]))
                values[input_tensor.name] = ca.MX.sym(input_tensor.name, total_size, 1)
            else:
                # Default to scalar if shape is unknown
                values[input_tensor.name] = ca.MX.sym(input_tensor.name)
                
        # Process initializers (weights, biases, constants)
        for initializer in model.graph.initializer:
            values[initializer.name] = self._convert_initializer(initializer)
            
        # Process each node in the graph
        for node in model.graph.node:
            output = self._convert_node(node, values)
            # Store the output(s) of this node
            for i, output_name in enumerate(node.output):
                if isinstance(output, (list, tuple)):
                    values[output_name] = output[i]
                else:
                    values[output_name] = output
                    
        # Return the output(s)
        outputs = []
        for output_tensor in model.graph.output:
            if output_tensor.name in values:
                outputs.append(values[output_tensor.name])
                
        # Return single output or vertcat of multiple outputs
        if len(outputs) == 1:
            return outputs[0]
        elif len(outputs) > 1:
            return ca.vertcat(*outputs)
        else:
            raise RuntimeError("No outputs found in the model")
            
    def _convert_node(self, node: NodeProto, values: Dict[str, ca.MX]) -> ca.MX:
        """
        Convert a single ONNX node to CasADi operations.
        
        Args:
            node (NodeProto): The ONNX node to convert
            values (Dict[str, ca.MX]): Dictionary of available values
            
        Returns:
            ca.MX: The resulting CasADi expression
            
        Raises:
            NotImplementedError: If the operation is not supported
        """
        op_type = node.op_type
        
        if op_type not in self.supported_ops:
            raise NotImplementedError(f"Operation '{op_type}' is not yet supported")
            
        # Get input values
        inputs = [values[input_name] for input_name in node.input if input_name in values]
        
        # Convert based on operation type
        if op_type == 'Add':
            return inputs[0] + inputs[1]
        elif op_type == 'Sub':
            return inputs[0] - inputs[1]
        elif op_type == 'Mul':
            return inputs[0] * inputs[1]
        elif op_type == 'Div':
            return inputs[0] / inputs[1]
        elif op_type == 'MatMul':
            return ca.mtimes(inputs[0], inputs[1])
        elif op_type == 'Gemm':
            # General matrix multiplication: Y = alpha * A * B + beta * C
            A, B = inputs[0], inputs[1]
            C = inputs[2] if len(inputs) > 2 else 0
            alpha = self._get_attribute(node, 'alpha', 1.0)
            beta = self._get_attribute(node, 'beta', 1.0)
            return alpha * ca.mtimes(A, B) + beta * C
        elif op_type == 'Relu':
            return ca.fmax(inputs[0], 0)
        elif op_type == 'Sigmoid':
            return 1 / (1 + ca.exp(-inputs[0]))
        elif op_type == 'Tanh':
            return ca.tanh(inputs[0])
        elif op_type == 'Identity':
            return inputs[0]
        else:
            # Placeholder for other operations
            raise NotImplementedError(f"Conversion for '{op_type}' not yet implemented")
            
    def _convert_initializer(self, initializer) -> ca.MX:
        """
        Convert an ONNX initializer (constant tensor) to CasADi.
        
        Args:
            initializer: ONNX initializer object
            
        Returns:
            ca.MX: CasADi constant matrix
        """
        # Convert ONNX tensor to numpy array
        import onnx.numpy_helper as numpy_helper
        numpy_array = numpy_helper.to_array(initializer)
        
        # Convert to CasADi MX
        return ca.MX(numpy_array)
        
    def _get_tensor_shape(self, tensor_info) -> List:
        """
        Extract shape information from a tensor.
        
        Args:
            tensor_info: ONNX tensor info object
            
        Returns:
            List: List of dimensions
        """
        if tensor_info.type.HasField('tensor_type'):
            if tensor_info.type.tensor_type.HasField('shape'):
                shape = []
                for dim in tensor_info.type.tensor_type.shape.dim:
                    if dim.HasField('dim_value'):
                        shape.append(dim.dim_value)
                    else:
                        shape.append(None)
                return shape
        return []
        
    def _get_attribute(self, node: NodeProto, name: str, default=None):
        """
        Get an attribute value from an ONNX node.
        
        Args:
            node (NodeProto): The ONNX node
            name (str): Attribute name
            default: Default value if attribute not found
            
        Returns:
            The attribute value or default
        """
        for attr in node.attribute:
            if attr.name == name:
                if attr.HasField('f'):
                    return attr.f
                elif attr.HasField('i'):
                    return attr.i
                elif attr.HasField('s'):
                    return attr.s
        return default
