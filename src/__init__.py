"""Instruct-Lite: A lightweight framework for instruction tuning and RAG."""

__version__ = "0.1.0"

# Make commonly used modules easily accessible
from . import model
from . import loader
from . import finetune
from . import formatter
from . import rag
from . import utils

__all__ = ["model", "loader", "finetune", "formatter", "rag", "utils"]

