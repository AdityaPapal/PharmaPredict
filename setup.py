from setuptools import setup,find_packages
from typing import List

HYPEN_E_DOT = '-e .'


def get_requirements(file_path:str)->List['str']:
    requirement = []
    
    with open(file_path) as f:
        requirement = f.readline()
        requirement = [req.replace('\n', "") for req in requirement]
        if HYPEN_E_DOT in requirement:
            requirement.remove(HYPEN_E_DOT)
        
        return requirement
        
setup(

    name= "PharmaSense",
    version= "0.0.1",
    author= "Aditya",
    author_email= "papaladitya@gmail.com",
    install_reqiure = get_requirements("requirements.txt"),
    packages= find_packages()
)


