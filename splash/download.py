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


def DownloadData(SpNumber, GAIAID="", user="", password=""):
    '''
    This function download Artemis data processed using cambridge pipeline from Cambridge server.

    Input
    ----------

    SpNumber:  string
            SPECULOOS target number such as SP0025+5422

    GAIAID:  integer
            GAIA ID corresponding to the SPECULOOS Target

    user: string
          Username to access the Cambridge data

    password: string
              password to access the Cambridge data
    '''

    if "TRAPPIST" in SpNumber:
        SpName = "Sp2306-0502"
        GAIAID = 2635476908753563008
    elif "TOI-736" in SpNumber:
        SpName = "TOI-736"
        GAIAID = 3562427951852172288
    elif not(GAIAID):
        SpName=SpNumber
        GAIAID =  GetID(SpNumber,IdType="SPECULOOS")
    else:
        SpName=SpNumber

    #Construct the path
    url = "http://www.mrao.cam.ac.uk/SPECULOOS/speculoos-portal/php/get_observations.php?id=%s&date=&filter=&telescope="  %GAIAID


    resp = requests.get(url, auth=(user, password))
    assert (
        resp.status_code == 200
    ), "Wrong username or password used to access data, please check it with Peter"
    assert (
        resp.content != b"null" and resp.content != b"\r\nnull"
    ), "Your request is not matching any available data in the Cambridge archive. To see available data, please check http://www.mrao.cam.ac.uk/SPECULOOS/portal_v2/"


    Content = eval(resp.content)[0]

    CompletePath = []
    DateValues = []
    FilterValues = []
    BaseLocation = "http://www.mrao.cam.ac.uk/SPECULOOS"

    for Item in Content:
        DateValues.append(Item['date'])
        FilterValues.append(Item['filter'])

        if "ARTEMIS" in Item['telescope'].upper() and  int(DateValues[-1])<20200927:
            version="v2_01"
        else:
            version="v2"
        ConstructedPath = os.path.join(BaseLocation,Item['telescope'],"output",version,DateValues[-1],SpName,"lightcurves")
        CompletePath.append(ConstructedPath)




    if len(DateValues)<1:
        print("Error downloading the data")
        return 0
    elif len(DateValues)>300:
        print("Too many data file found")
        return 0
    else:
        print("Found %d different nights of observation. Downloading now." %len(DateValues))
    #Clean the TempFolder
    os.system("rm -rf TempFolder/*")

    for Path, Filter, Date in tqdm(zip(CompletePath, FilterValues, DateValues)):

        if not(os.path.exists("TempFolder")):
            os.system("mkdir TempFolder")

        urlGet3 = os.path.join(Path, "%s_%s_%s_3_MCMC" %(GAIAID, Filter, Date))
        urlGet4 = os.path.join(Path, "%s_%s_%s_4_MCMC" %(GAIAID, Filter, Date))
        urlGet5 = os.path.join(Path, "%s_%s_%s_5_MCMC" %(GAIAID, Filter, Date))
        urlGet6 = os.path.join(Path, "%s_%s_%s_6_MCMC" %(GAIAID, Filter, Date))
        urlGet7 = os.path.join(Path, "%s_%s_%s_7_MCMC" %(GAIAID, Filter, Date))
        urlGet8 = os.path.join(Path, "%s_%s_%s_8_MCMC" %(GAIAID, Filter, Date))

        rGET3 = requests.get(urlGet3, auth=(user, password))
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


        if not("TITLE" in rGET3.text.upper()):
            with open(SaveFileName3,'w') as f:
                f.write(rGET3.text)

        if not("TITLE" in rGET4.text.upper()):
            with open(SaveFileName4,'w') as f:
                f.write(rGET4.text)

        if not("TITLE" in rGET5.text.upper()):
            with open(SaveFileName5,'w') as f:
                f.write(rGET5.text)

        if not("TITLE" in rGET6.text.upper()):
            with open(SaveFileName6,'w') as f:
                f.write(rGET6.text)

        if not("TITLE" in rGET7.text.upper()):
            with open(SaveFileName7,'w') as f:
                f.write(rGET7.text)

        if not("TITLE" in rGET8.text.upper()):
            with open(SaveFileName8,'w') as f:
                f.write(rGET8.text)

    CombineData(SpNumber)
    return 1


def CombineData(SpNumber):
    '''
    Combines the data in the TempFolder when downloaded
    '''

    Parameters= "BJDMID, FLUX, DX, DY, FWHM, FWHM_X, FWHM_Y, SKY, AIRMASS"
    for Aper in range(3,9):
        CurrentFileList = glob.glob("TempFolder/*ap%s.txt" %Aper)
        if len(CurrentFileList)<1:
            continue
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
        if not(os.path.exists("data")):
            os.system("mkdir data")

        np.savetxt("data/%s_%sAp.txt" %(SpNumber, Aper), AllData, header=Parameters)

    os.system("rm -rf TempFolder")




def DownloadFitsData(SpNumber, GAIAID="", user="", password=""):
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

    if "TRAPPIST-1" in SpNumber:
        SpName = "Sp2306-0502"
        GAIAID = 2635476908753563008
    elif "TOI-736" in SpNumber:
        SpName = "TOI-736"
        GAIAID = 3562427951852172288
    else:
        SpName = SpNumber
        GAIAID =  GetID(SpNumber,IdType="SPECULOOS")

    #Construct the path
    url = "http://www.mrao.cam.ac.uk/SPECULOOS/speculoos-portal/php/get_observations.php?id=%s&date=&filter=&telescope="  %GAIAID



    resp = requests.get(url, auth=(user, password))
    assert (
        resp.status_code == 200
    ), "Wrong username or password used to access data, please check it with Peter"
    assert (
        resp.content != b"null" and resp.content != b"\r\nnull"
    ), "Your request is not matching any available data in the Cambridge archive. To see available data, please check http://www.mrao.cam.ac.uk/SPECULOOS/portal_v2/"

    print("The value of GAIAID is:", GAIAID)


    Content = eval(resp.content)[0]
    CompletePath = []
    DateValues = []
    FilterValues = []
    BaseLocation = "http://www.mrao.cam.ac.uk/SPECULOOS"


    if len(Content)<1:
        print("Error downloading the data")
        return 0

    for Item in Content:
        DateValues.append(Item['date'])
        FilterValues.append(Item['filter'])

        if "ARTEMIS" in Item['telescope'].upper() and  int(DateValues[-1])<20200927:
            version="v2_01"
        else:
            version="v2"
        ConstructedPath = os.path.join(BaseLocation,Item['telescope'],"output",version,DateValues[-1],SpName)
        CompletePath.append(ConstructedPath)


    os.system("rm -rf TempFolder/*")

    for Path, Filter, Date in zip(CompletePath, FilterValues, DateValues):

        print("Downloading Date:", Date)
        if not(os.path.exists("TempFolder")):
            os.system("mkdir TempFolder")
        if not(os.path.exists("TempFolder/%s" %str(SpName))):
            os.system("mkdir TempFolder/%s" %str(SpName))

        Catalogue = Path+"/1_initial-catalogue.fits"
        urlGet3 = Path+"/%s_%s_%s_3_diff.fits" %(GAIAID, Filter, Date)
        urlGet4 = Path+"/%s_%s_%s_4_diff.fits" %(GAIAID, Filter, Date)
        urlGet5 = Path+"/%s_%s_%s_5_diff.fits" %(GAIAID, Filter, Date)
        urlGet6 = Path+"/%s_%s_%s_6_diff.fits" %(GAIAID, Filter, Date)
        urlGet7 = Path+"/%s_%s_%s_7_diff.fits" %(GAIAID, Filter, Date)
        urlGet8 = Path+"/%s_%s_%s_8_diff.fits" %(GAIAID, Filter, Date)


        rGET3 = requests.get(urlGet3, auth=(user, password))
        rGET4 = requests.get(urlGet4, auth=(user, password))
        rGET5 = requests.get(urlGet5, auth=(user, password))
        rGET6 = requests.get(urlGet6, auth=(user, password))
        rGET7 = requests.get(urlGet7, auth=(user, password))
        rGET8 = requests.get(urlGet8, auth=(user, password))

        SaveFileName0 = "TempFolder/%s/Catalogue.fits" %str(SpNumber)
        SaveFileName3 = "TempFolder/%s/%s_%s_SPC_3.fits" %(str(SpNumber), Filter, Date)
        SaveFileName4 = "TempFolder/%s/%s_%s_SPC_4.fits" %(str(SpNumber), Filter, Date)
        SaveFileName5 = "TempFolder/%s/%s_%s_SPC_5.fits" %(str(SpNumber), Filter, Date)
        SaveFileName6 = "TempFolder/%s/%s_%s_SPC_6.fits" %(str(SpNumber), Filter, Date)
        SaveFileName7 = "TempFolder/%s/%s_%s_SPC_7.fits" %(str(SpNumber), Filter, Date)
        SaveFileName8 = "TempFolder/%s/%s_%s_SPC_8.fits" %(str(SpNumber), Filter, Date)

        if not(os.path.exists(SaveFileName0)):
            rGETCat = requests.get(Catalogue, auth=(user, password))
            with open(SaveFileName0,'w') as f:
                if len(rGETCat.text)>200:
                    f.write(rGETCat.text)

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
