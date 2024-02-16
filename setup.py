from setuptools import setup

setup(
    name='DRUID',
    version='0.0.0',
    author='Rhys Shaw',
    author_email='rhys.shaw@bristol.ac.uk',
    url='https://github.com/RhysAlfShaw/DRUID',
    description='descriptions',
    packages=['DRUID',
            'DRUID/src',
            'DRUID/src/background',
            'DRUID/src/homology',
            'DRUID/src/source',
            'DRUID/src/utils'],
)