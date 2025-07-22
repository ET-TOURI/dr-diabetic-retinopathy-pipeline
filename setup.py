from setuptools import setup, find_packages

setup(
    name='dr_classification',
    version='0.1.0',
    author=['Abdelhak'
            'abdelhakettouri@gmail.com'
            ],
    description='Pipeline modulaire pour la classification de la rétinopathie diabétique avec PSO, Grad-CAM et Streamlit',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ET-TOURI/dr-diabetic-retinopathy-pipeline',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'tensorflow',
        'opencv-python',
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'pyswarm',
        'streamlit'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
)