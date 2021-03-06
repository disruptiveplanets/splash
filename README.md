
# <img src="logo/logo.png"  alt="SPLASH"  style="float: left; margin-right: 10px;" />


## Developers
Prajwal Niraula, Lionel Garcia, Artem Burdanov, Benjamin Rackham

## Email:
prajwalniraula@gmail.com, lionel.garcia@student.uliege.be

## Additional Team Members:
Julien de Wit

# Description
SPLASH (Speculoos PLAnet SearcH) is a module that works with cambridge products in order to detect planets. SPLASH is powered by SVD, and it finds the best basis vector that explains the trend in data while preserving box like transit shapes. Current version is meant to work with old/new version of SPECULOOS Cambridge Pipeline products. Also it is meant to work with TRAPPHOT products. SPLASH also utilizes multiprocessing if resources are available to improve the speed of the algorithm.

# Installation Guide:

## Option 1: Install from GitHub via `pip`
From the command line, simply type:

    pip install git+https://github.com/disruptiveplanets/splash

## Option 2: Install from Source
To install, first clone the code:

    git clone https://github.com/disruptiveplanets/splash

Then, go the the splash folder by:

    cd splash

Install by typing in the following command:

    python setup.py install

You may require superuser privileges:

    sudo python setup.py install

# Products:
The stable version of the code resides at https://github.com/disruptiveplanets/splash, whereas the code is actively being developed is at https://github.com/prajwal309/Splash

SPLASH creates following subfolders within the output directory:
- DailyBestCases: Diagnostic plots for  containing the best case fit for each night.
- Data: contains the pickled data from each run. This data might be used in the future code to start off MCMC fitting.
- Outliers: If data is pre-cleaned, which is recommended, outliers are shown for each night in this folder.
- Periodogram: Periodograms from either "TLS" or "TransitMatching" algorithm. More algorithms currently under development.
- MCMC_Results: If MCMC results are run, resulting fits results are located here.


# Future development:
- Algorithm are constantly being improved. A more robust and faster transit matching algorithm is being build, and currently being tested.
- A completeness test will be performed using the algorithm, and perhaps part of the code.


# Example:
A simple example to download the data is - login details required from Peter:
from splash.download import DownloadData
DownloadData(Target,user = "....",password = "....")
