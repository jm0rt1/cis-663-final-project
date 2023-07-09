import sys
from setuptools import setup, find_packages

# Dependencies are automatically detected, but it might need fine tuning.
# "packages": ["os"] is used as example only
build_exe_options = {"packages": ["os"], "excludes": ["tkinter"]}

# base="Win32GUI" should be used only for Windows GUI app
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="{project-name}",
    version="0.1",
    packages=find_packages(),
    package_data={'{project-name}': ['scripts/*']},
    include_package_data=True,
    description="",
    install_requires=open('requirements.txt').readlines(),
    author="",
    autho_email="",
    url=""
)
