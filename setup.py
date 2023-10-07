from setuptools import setup, find_packages

setup(
    name="AdaptableNeuralNetwork",
    version="0.1",
    author="Shuai Li",
    author_email="shuailirpi@gmail.com",
    description="Implementation of adaptable neural network",
    #long_description="Detailed description of your package",
    #url="https://github.com/yourusername/my_package",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        # List your package dependencies here
    ],
)
