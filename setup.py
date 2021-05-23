import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mypgm", # Replace with your own username
    version="0.0.1",
    author="PIERO_CIFFOLILLO",
    author_email="pierociffolillo@gmail.com",
    description="A package for pgm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pierociffo",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
