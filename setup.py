# Professional MLOps Project Setup
from setuptools import setup, find_packages

# Read requirements from requirements.txt
def read_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read long description from README
def read_long_description():
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "MLOps Reproducible Pipeline for Obesity Classification"

setup(
    name="mlops-obesity-classifier",
    version="1.0.0",
    description="Production-ready MLOps pipeline for obesity classification with comprehensive experiment tracking",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    
    # Author information
    author="Alicia Cantarero",
    author_email="alicia.cantarero@example.com",
    
    # Project URLs
    url="https://github.com/yourusername/mlops-reproducible",
    project_urls={
        "Documentation": "https://github.com/yourusername/mlops-reproducible/docs",
        "Source": "https://github.com/yourusername/mlops-reproducible",
        "Tracker": "https://github.com/yourusername/mlops-reproducible/issues",
    },
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Optional dependencies for development
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
            "pre-commit>=3.0",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.18",
        ],
        "monitoring": [
            "prometheus-client>=0.16",
            "grafana-api>=1.0",
        ],
        "deployment": [
            "docker>=6.0",
            "kubernetes>=26.0",
        ]
    },
    
    # Entry points for CLI commands
    entry_points={
        "console_scripts": [
            "mlops-train=src.scripts.train:main",
            "mlops-predict=src.scripts.predict:main",
            "mlops-evaluate=src.scripts.evaluate:main",
            "mlops-monitor=src.scripts.monitor:main",
        ],
    },
    
    # Package classification
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Keywords for discoverability
    keywords=[
        "mlops", "machine-learning", "obesity-classification", 
        "healthcare", "mlflow", "dvc", "scikit-learn",
        "production-ml", "model-registry", "experiment-tracking"
    ],
    
    # Package data
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md"],
    },
    
    # Data files
    data_files=[
        ("configs", ["params.yaml", "dvc.yaml", "mlflow_standards.yaml"]),
        ("docs", ["README.md", "CHANGELOG.md"]),
    ],
    
    # Zip safe
    zip_safe=False,
    
    # License
    license="MIT",
)