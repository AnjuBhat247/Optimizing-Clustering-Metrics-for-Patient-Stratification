from setuptools import setup, find_packages

setup(
    name='ClusteringAlgorithms',  
    version='0.1.0',  # Initial version number
    packages=find_packages(),
    install_requires=[
        'numpy',        # List of dependencies
        'pandas',
        'scikit-learn',
        'matplotlib',
        'snfpy',
        'scikit-survival',
        'lifelines',
        'scipy'
    ],
    author='Anjana BHAT',
    author_email='bhatanjana.ab@gmail.com',  
    description='Clustering Algorithms for Patient Stratification',  
    long_description=open('README.md').read(),  
    long_description_content_type='text/markdown',
    url='https://github.com/AnjuBhat247/Optimizing-Clustering-Metrics-for-Patient-Stratification.git',  
    python_requires='>=3.10', 
)
