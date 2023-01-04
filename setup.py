from setuptools import setup
from os import path


# The long description comes from the readme.
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyoptDMD',
    version='0.1',
    # packages=['src', 'tests'],
    url='https://github.com/klapo/pyoptDMD',
    license='MIT License',
    author='Karl Lapo,  James Kunert-Graf, Travis Askham',
    author_email='karl-eric.lapo@uibk.ac.at',
    description='Python implementation of the optimized variable projection dynamic '
                'mode decomposition method. ',
    package_dir={'': 'src'},
    long_description=long_description,
    python_requires='>=3.8',
    install_requires=[
        'munkres>=1.1.4',
        'scipy>=1.9.1',
        'numpy>=1.23.4',
        'setuptools>=65.5.1',
    ],

)
