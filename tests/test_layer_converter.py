"""
Tests for the layer converter module.
"""
import pytest
from onnx2casadi.layers.layer_converter import LayerConverter


class TestLayerConverter:
    """Test cases for LayerConverter class."""
    
    def test_initialization(self):
        """Test that LayerConverter can be initialized."""
        converter = LayerConverter()
        assert converter is not None
        assert len(converter.supported_ops) > 0
        
    def test_supported_operations(self):
        """Test that basic operations are supported."""
        converter = LayerConverter()
        
        # Check some basic operations
        assert 'Add' in converter.supported_ops
        assert 'Mul' in converter.supported_ops
        assert 'MatMul' in converter.supported_ops
        assert 'Relu' in converter.supported_ops
