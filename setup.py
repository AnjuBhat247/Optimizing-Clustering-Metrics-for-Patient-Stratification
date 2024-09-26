from setuptools import setup, find_packages

setup(
    name='ClusteringAlgorithms',  
    version='0.1.0',  # Initial version number
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',          
        'pandas>=1.3.0',          
        'scikit-learn>=0.24.0',   
        'matplotlib>=3.4.0',     
        'snfpy>=0.2.0',           
        'scikit-survival>=0.16.0',
        'lifelines>=0.26.0',      
        'scipy>=1.7.0',          
        'pygad>=2.17.0'
    ],
    author='Anjana BHAT',
    author_email='bhatanjana.ab@gmail.com',  
    description='Clustering Algorithms for Patient Stratification',  
    long_description=open('README.md').read(),  
    long_description_content_type='text/markdown',
    url='https://github.com/AnjuBhat247/Optimizing-Clustering-Metrics-for-Patient-Stratification.git',  
    python_requires='>=3.10', 
)
