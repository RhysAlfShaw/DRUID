from setuptools import setup

with open('requirements.txt') as f:
    requiremets = f.read().splitlines()

setup(
    name='DRUID',
    version='0.0.0',
    author='Rhys Shaw',
    author_email='rhys.shaw@bristol.ac.uk',
    url='https://github.com/RhysAlfShaw/DRUID',
    description='descriptions',
    install_requires=requiremets,

    packages=['DRUID',
            'DRUID/src'],
)