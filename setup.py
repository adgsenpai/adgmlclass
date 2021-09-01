from setuptools import setup, find_packages

setup(
    name="adgmlclass",
    version="0.0.1",
    author="Ashlin Darius Govindasamy",
    author_email="adgrules@hotmail.com",
    url="https://www.adgstudios.co.za",
    description="A class used for finding the best machine learning algorithm",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["click", "pytz","pandas","numpy","matplotlib","seaborn","scikit-learn"],
    entry_points={"console_scripts": ["app = src.app:main"]},
    
)
