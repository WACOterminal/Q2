from setuptools import setup, find_packages

setup(
    name='ml_event_processor',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'apache-flink==1.17.0',
        'apache-pulsar==2.11.0',
        'elasticsearch==8.11.0',
        'apache-avro',
        'pyyaml'
    ],
    python_requires='>=3.8',
) 