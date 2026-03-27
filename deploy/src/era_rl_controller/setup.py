import os
from glob import glob

from setuptools import find_packages, setup

package_name = "era_rl_controller"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "configs"), glob("configs/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Guangxiao Yang",
    maintainer_email="guangxiao.yang@robotera.com",
    description="TODO: Package description",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "era_rl_controller_node = era_rl_controller.era_rl_controller_node:main",
        ],
    },
)
