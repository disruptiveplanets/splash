import os
import requests

import numpy as np
from os import path
from tqdm import tqdm
from lxml import etree
from astropy.io import fits
from astropy.io import fits
from io import BytesIO

def get_file_from_url(url, user, password):
    resp = requests.get(url, auth=(user, password))
    return BytesIO(resp.content)


def cambridge_download_product(
    destination,
    user,
    password,
    target="",
    date="",
    telescope="",
    stack="global",
    overwrite=False,
):
    """
    Function written by Peter Pedersen.

    Download photometry products from cambridge archive in the following file tree:

    .. code-block::

        Target/
         │
         └── Telescope_date_target_filter/
              ├── Telescope_date_target_filter_photometry.phots
              └── Telescope_date_target_filter_stack.fits

    Parameters
    ----------
    destination: string (path)
        local destination path
    target : string
        target name in the cambridge archive
    date : string
        date with format YYYYmmdd (e.g. 20190130)
    telescope : string
        telescope name in the cambridge archive
    process_lc: bool
        whether to process the lighturve and store it in the ``phots``file
    user : string
        username for the cambridge archive
    password : string
        password for the cambridge archive
    target_list_path : string path
        local path of the target list file (csv)
    target_gaia_id : string
        gaia id of the target (optional, will be found from the target list)

    """
    destination = path.abspath(destination)

    if not path.exists(destination):
        print("Creating destination path:", destination)
        os.mkdir(destination)

    url = "http://www.mrao.cam.ac.uk/SPECULOOS/portal/get_file.php?telescope={}&date={}&id={}&filter=&file=../../*.fits".format(
        telescope, date, target
    )

    resp = requests.get(url, auth=(user, password))
    assert (
        resp.status_code == 200
    ), "Wrong username or password used to access data, please check .specphot.config file"
    assert (
        resp.content != b"null" and resp.content != b"\r\nnull"
    ), "Your request is not matching any available data in the Cambridge archive. To see available data, please check http://www.mrao.cam.ac.uk/SPECULOOS/portal/"
    output_fits_urls = np.array(
        [
            ("http://www.mrao.cam.ac.uk/SPECULOOS/" + ur[4::]).replace("\\", "")
            for ur in eval(resp.content)
        ]
    )


    for url in tqdm(output_fits_urls):
        if "output" in url:
            target_name = url.split("/")[7]
            telesope = url.split("/")[4]
            date = url.split("/")[6]
            filter = url.split("/")[9]

            denominator = "{}_{}_{}_{}".format(telesope, date, target_name, filter)


            if not path.exists("data"):
                os.mkdir("data")
            date_folder = os.path.join("data", target_name)



            if not path.exists(date_folder):
                os.mkdir(date_folder)

            product_path = path.join(date_folder, "{}_output.fits".format(denominator))


            fits.open(get_file_from_url(url, user, password)).writeto(product_path, overwrite=True)
