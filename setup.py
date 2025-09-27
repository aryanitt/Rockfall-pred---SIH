from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    Reads dependencies from requirements.txt and returns them as a list.
    Removes '-e .' if present.
    """
    requirements = []
    with open(file_path) as file_obj:
        for line in file_obj:
            line = line.strip()
            if line and line != HYPEN_E_DOT:
                requirements.append(line)
    return requirements


setup(
    name="rockfall_risk_predictor",
    version="0.1.0",
    description="A machine learning project for real-time rockfall risk prediction using sensor data.",
    author="Aryan Gupta",
    author_email="aryanguptaofficial98@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)
