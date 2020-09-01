import os
import requests
import glob
import numpy as np
from os import path
from tqdm import tqdm
from astropy.io import fits
from astropy.io import fits
from io import BytesIO
from .Functions import GetID

def get_file_from_url(url, user, password):
    resp = requests.get(url, auth=(user, password))
    return BytesIO(resp.content)


def DownloadData(SpNumber, user="", password=""):
    '''
    This function download Artemis data processed using cambridge pipeline from Cambridge server.

    Input
    ----------

    SpNumber:  string
                SPECULOOS target number such as SP0025+5422

    user: string
          Username to access the Cambridge data

    password: string
              password to access the Cambridge data
    '''

    GAIAID =  GetID(SpNumber,IdType="SPECULOOS")

    #Construct the path
    url = "http://www.mrao.cam.ac.uk/SPECULOOS/portal_v2/php/get_dir.php?id=%s&date=&filter=&telescope="  %GAIAID



    resp = requests.get(url, auth=(user, password))
    assert (
        resp.status_code == 200
    ), "Wrong username or password used to access data, please check .specphot.config file"
    assert (
        resp.content != b"null" and resp.content != b"\r\nnull"
    ), "Your request is not matching any available data in the Cambridge archive. To see available data, please check http://www.mrao.cam.ac.uk/SPECULOOS/portal_v2/"


    Content = eval(resp.content)

    Directories = Content['dirs']
    DateValues = Content['date_vals']
    FilterValues = Content['filter_vals']
    BaseLocation = "http://www.mrao.cam.ac.uk/SPECULOOS"

    CompletePath = []
    for Directory in Directories:
        ConstructedPath = BaseLocation+Directory[6:].replace("\\","")+"lightcurves/"
        CompletePath.append(ConstructedPath)

    #Clean the TempFolder
    os.system("rm TempFolder/*")

    for Path, Filter, Date in zip(CompletePath, FilterValues, DateValues):

        if not(os.path.exists("TempFolder")):
            os.system("mkdir TempFolder")

        urlGet3 = Path+"%s_%s_%s_3_MCMC" %(GAIAID, Filter, Date)
        urlGet4 = Path+"%s_%s_%s_4_MCMC" %(GAIAID, Filter, Date)
        urlGet5 = Path+"%s_%s_%s_5_MCMC" %(GAIAID, Filter, Date)
        urlGet6 = Path+"%s_%s_%s_6_MCMC" %(GAIAID, Filter, Date)
        urlGet7 = Path+"%s_%s_%s_7_MCMC" %(GAIAID, Filter, Date)
        urlGet8 = Path+"%s_%s_%s_8_MCMC" %(GAIAID, Filter, Date)

        rGET3 = requests.get(urlGet4, auth=(user, password))
        rGET4 = requests.get(urlGet4, auth=(user, password))
        rGET5 = requests.get(urlGet5, auth=(user, password))
        rGET6 = requests.get(urlGet6, auth=(user, password))
        rGET7 = requests.get(urlGet7, auth=(user, password))
        rGET8 = requests.get(urlGet8, auth=(user, password))

        SaveFileName3 = "TempFolder/%s_%s_SPC_ap3.txt" %(str(SpNumber), Date)
        SaveFileName4 = "TempFolder/%s_%s_SPC_ap4.txt" %(str(SpNumber), Date)
        SaveFileName5 = "TempFolder/%s_%s_SPC_ap5.txt" %(str(SpNumber), Date)
        SaveFileName6 = "TempFolder/%s_%s_SPC_ap6.txt" %(str(SpNumber), Date)
        SaveFileName7 = "TempFolder/%s_%s_SPC_ap7.txt" %(str(SpNumber), Date)
        SaveFileName8 = "TempFolder/%s_%s_SPC_ap8.txt" %(str(SpNumber), Date)


        with open(SaveFileName3,'w') as f:
            if len(rGET3.text)>200:
                f.write(rGET3.text)

        with open(SaveFileName4,'w') as f:
            if len(rGET4.text)>200:
                f.write(rGET4.text)

        with open(SaveFileName5,'w') as f:
            if len(rGET5.text)>200:
                f.write(rGET5.text)

        with open(SaveFileName6,'w') as f:
            if len(rGET6.text)>200:
                f.write(rGET6.text)

        with open(SaveFileName7,'w') as f:
            if len(rGET7.text)>200:
                f.write(rGET7.text)

        with open(SaveFileName8,'w') as f:
            if len(rGET8.text)>200:
                f.write(rGET8.text)

        print("data Saved File for %s" %Date)

    print("Now combining data to a single file")
    CombineData(SpNumber)


def CombineData(SpNumber):
    '''
    Combines the data in the TempFolder when downloaded
    '''

    Parameters= "BJDMID, FLUX, DX, DY, FWHM, FWHM_X, FWHM_Y, SKY, AIRMASS"
    for Aper in range(3,9):
        CurrentFileList = glob.glob("TempFolder/*ap%s.txt" %Aper)

        AllData = []
        for FileItem in CurrentFileList:
            try:
                DataText = np.genfromtxt(FileItem, skip_header=1)
                X,Y = np.shape(DataText)

                CurrentData = np.empty((X, 9))
                CurrentData[:,0] =  DataText[:,1]
            except:
                continue
            CurrentData[:,1] =  DataText[:,3]
            CurrentData[:,2] =  DataText[:,6]
            CurrentData[:,3] =  DataText[:,7]
            CurrentData[:,4] =  DataText[:,8]
            CurrentData[:,5] =  DataText[:,9]
            CurrentData[:,6] =  DataText[:,10]
            CurrentData[:,7] =  DataText[:,11]
            CurrentData[:,8] =  DataText[:,12]
            AllData.extend(CurrentData)
        AllData = np.array(AllData)
        AllTime = AllData[:,0]
        AllData = AllData[np.argsort(AllTime)]
        if os.path.exists("data"):
            print("Saving inside data")
            np.savetxt("data/%s_%sAp.txt" %(SpNumber, Aper), AllData, header=Parameters)
        else:
            print("Saving at the current directory")
            np.savetxt("%s_%sAp.txt" %(SpNumber, Aper), AllData, header=Parameters)

    os.system("rm -rf TempFolder")
