from setuptools import find_packages
from setuptools import setup
REQUIRED_PACKAGES = ['natsort>=8.0.2', "Pillow==8.4.0"]
setup(
    name='universal_nst',
    version='0.1',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    description='My training application.'
)
