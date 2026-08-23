from setuptools import find_packages, setup
from typing import List

#-------------------------------------------------------Requirements function to fetch requirements from requirements.txt
e_dot = '-e .'

def get_requirements(file_path: str) -> List[str]:
    
    '''This function will return the list of requirements'''
    
    requirements = []
    
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        if e_dot in requirements:
            requirements.remove(e_dot)
            
        return requirements

setup(
    name = 'Attendance System',
    version = '1.0.0',
    description = 'Manual attendance in large colleges consumes valuable classroom time and is prone to proxy attendance, reducing effective teaching hours.',
    author = 'Scanova',
    author_email = 'debashisxmajumder@gmail.com',
    package_dir = {"": "src"},
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt'),
)