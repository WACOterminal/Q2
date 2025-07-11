# shared/opentelemetry/setup.py
from setuptools import setup, find_packages

setup(
    name="q_opentelemetry",
    version="0.1.0",
    packages=find_packages(),
    description="A shared library for OpenTelemetry configuration in the Q platform.",
    author="Q Platform Development Team",
    author_email="dev@q-platform.ai",
) 