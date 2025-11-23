"""
Tests for the ONNX importer module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from onnx2casadi.importers.onnx_importer import OnnxImporter


class TestOnnxImporter:
    """Test cases for OnnxImporter class."""
    
    def test_initialization(self):
        """Test that OnnxImporter can be initialized."""
        importer = OnnxImporter()
        assert importer is not None
        
    def test_load_nonexistent_file_raises_error(self):
        """Test that loading a non-existent file raises FileNotFoundError."""
        importer = OnnxImporter()
        
        with pytest.raises(FileNotFoundError):
            importer.load_model("nonexistent_file.onnx")
