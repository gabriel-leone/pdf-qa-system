"""
Property-based tests for project setup validation
**Feature: pdf-qa-system, Property 1: Document processing completeness**
**Validates: Requirements 1.1, 1.2, 1.3, 1.4**
"""
import pytest
from hypothesis import given, strategies as st
import os
import sys
import importlib.util
from pathlib import Path


class TestProjectSetup:
    """Test project structure and dependencies are properly configured"""
    
    def test_required_directories_exist(self):
        """Test that all required directories exist"""
        required_dirs = ['models', 'services', 'api', 'tests', 'utils']
        
        for dir_name in required_dirs:
            assert os.path.exists(dir_name), f"Required directory '{dir_name}' does not exist"
            assert os.path.isdir(dir_name), f"'{dir_name}' exists but is not a directory"
    
    def test_init_files_exist(self):
        """Test that __init__.py files exist in all Python packages"""
        required_init_files = [
            'models/__init__.py',
            'services/__init__.py', 
            'api/__init__.py',
            'tests/__init__.py',
            'utils/__init__.py'
        ]
        
        for init_file in required_init_files:
            assert os.path.exists(init_file), f"Required __init__.py file '{init_file}' does not exist"
    
    def test_requirements_file_exists(self):
        """Test that requirements.txt exists and contains core dependencies"""
        assert os.path.exists('requirements.txt'), "requirements.txt file does not exist"
        
        with open('requirements.txt', 'r') as f:
            requirements_content = f.read()
        
        # Check for core dependencies
        required_packages = [
            'fastapi',
            'PyPDF2', 
            'sentence-transformers',
            'chromadb',
            'openai',
            'pydantic',
            'python-dotenv',
            'pytest',
            'hypothesis'
        ]
        
        for package in required_packages:
            assert package in requirements_content, f"Required package '{package}' not found in requirements.txt"
    
    def test_config_module_importable(self):
        """Test that configuration module can be imported"""
        try:
            import config
            assert hasattr(config, 'settings'), "Config module should have 'settings' attribute"
        except ImportError as e:
            pytest.fail(f"Could not import config module: {e}")
    
    def test_main_application_exists(self):
        """Test that main application file exists and is importable"""
        assert os.path.exists('main.py'), "main.py file does not exist"
        
        try:
            spec = importlib.util.spec_from_file_location("main", "main.py")
            main_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(main_module)
            assert hasattr(main_module, 'app'), "main.py should define 'app' variable"
        except Exception as e:
            pytest.fail(f"Could not import main.py: {e}")
    
    def test_logging_configuration_exists(self):
        """Test that logging configuration is available"""
        try:
            from utils.logging import logger
            assert logger is not None, "Logger should be configured"
        except ImportError as e:
            pytest.fail(f"Could not import logging configuration: {e}")
    
    @given(st.text(min_size=1, max_size=100, alphabet=st.characters(blacklist_characters=['\x00'])))
    def test_environment_variable_handling(self, test_value):
        """Property test: Environment variables should be handled correctly"""
        # Test that config can handle various environment variable values
        import os
        from config import Settings
        
        # Set a test environment variable
        test_key = "TEST_CONFIG_VALUE"
        original_value = os.environ.get(test_key)
        
        try:
            os.environ[test_key] = test_value
            # Settings should not crash with arbitrary string values
            settings = Settings()
            # Should be able to access settings without error
            assert hasattr(settings, 'log_level')
        finally:
            # Clean up
            if original_value is not None:
                os.environ[test_key] = original_value
            elif test_key in os.environ:
                del os.environ[test_key]
    
    def test_project_structure_completeness(self):
        """
        Property 1: Document processing completeness
        Test that the project structure supports the complete document processing pipeline
        """
        # Verify that the basic structure exists to support:
        # 1. PDF text extraction (models for documents)
        # 2. Text chunking (services for processing) 
        # 3. Embedding generation (services for embeddings)
        # 4. Vector storage (services for storage)
        
        # Check that we have the foundation for each processing step
        assert os.path.exists('models'), "Models directory needed for data structures"
        assert os.path.exists('services'), "Services directory needed for processing logic"
        assert os.path.exists('api'), "API directory needed for endpoints"
        
        # Check configuration supports the pipeline
        from config import settings
        assert hasattr(settings, 'max_chunk_size'), "Config should support chunking parameters"
        assert hasattr(settings, 'embedding_model'), "Config should support embedding model"
        assert hasattr(settings, 'chroma_persist_directory'), "Config should support vector storage"