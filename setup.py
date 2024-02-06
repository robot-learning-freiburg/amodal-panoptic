from setuptools import setup, find_packages

setup(
    name='amodal_panoptic_eval',
    version='1.0.0',
    description='Amodal Panoptic Segmentation Evaluation',
    author='Rohit Mohan',
    author_email='mohan@cs.uni-freiburg.de',
    url='https://github.com/robot-learning-freiburg/amodal-panoptic.git',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'pycocotools',
        'tqdm',
        'scipy',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
