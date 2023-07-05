import setuptools

setuptools.setup(
    name="pyfrechet",
    version="1.0.0",
    author="Matthieu Bulté",
    author_email="mb@math.ku.dk",
    description="A module for the manipulation and analysis of data in metric spaces.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/matthieubulte/pyfrechet",
    project_urls={
        "Bug Tracker": "https://github.com/matthieubulte/pyfrechet/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(include="pyfrechet*"),
    python_requires=">=3.9",
    install_requires=[
        'fdasrsf>=2.3.12',
        'geomstats>=2.5.0',
        'joblib>=1.2.0',
        'numpy>=1.23.5',
        'pandas>=1.5.3',
        'pytest>=7.2.2',
        'requests>=2.31.0',
        'scikit_fda>=0.8',
        'scikit_learn>=1.2.2',
        'scipy>=1.9.1',
        'tqdm>=4.64.1'
    ]
)