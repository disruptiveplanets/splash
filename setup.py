import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="splash",
    version="0.1",
    author="Prajwal Niraula and Lionel Garcia",
    author_email="prajwalniraula@gmail.com/garcia@uliege.be",
    description="Splash is a target to look for transiting exoplanets in SPECULOOS database.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/prajwal309/splash",
    packages=setuptools.find_packages(),
    include_package_data = True,


    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=[
        'setuptools>=46.4.0'
        'astropy>3',  # astropy 3 doesn't install in Python 2, but is req for astroquery
        'matplotlib>=2.0.0',  # earlier has bug for "from astroquery.mast import Catalogs"
        'numpy',
        'scipy',
        'tqdm',
        'batman-package',
        'argparse',
        'emcee',
        'numdifftools',
        'lmfit',
        'transitleastsquares',
     ]
)
