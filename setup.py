from setuptools import setup, find_packages

# 读取 requirements.txt 文件
with open('requirements_deploy.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='light_mappo',
    version='1.0.0',
    packages=find_packages(),
    install_requires=requirements,
)
