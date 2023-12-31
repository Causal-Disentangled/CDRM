import setuptools
with open("README.md", "r") as fh:  
    long_description = fh.read()
setuptools.setup(
    name="causal_disentangle_pkg", 
    version="0.0.1",   
    author="mengyueyang",  
    author_email="yangmengyue2@huawei.com",
    description="Causal toy images and disentanglement method",
    long_description=long_description,     
    long_description_content_type="text/markdown",  
    url="*",  
    packages=setuptools.find_packages(), 
    
    classifiers=[   
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  
)