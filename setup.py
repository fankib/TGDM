import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tgdm", # Replace with your own username
    version="0.1",
    author="Benjamin Fankhauser",
    author_email="nimajneb_fankhauser@hotmail.com",
    description="Tuned Gradient Descent with Momentum",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fankib/TGDM",
    packages=setuptools.find_packages(exclude=['experiment']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)