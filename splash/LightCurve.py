import numpy as np
from .Functions import ReadData, ParseFile
from time import time
import os


class Target:
    '''
    This is a class for a target of data
    '''

    def __init__(self, Location="data", Name ="", Output=""):
        '''
        Expect a continuous data concatenated data with header in the first row.
        First row is expected to be time. Second row is expected to be flux.

        ParamName: The name of the parameters.
        ParamValue: The values of the parameters.
        '''

        if len(Location)>0 and len(Name)>0:
            self.ParamNames, self.ParamValues = ReadData(Location, Name)

        #Generate the output folder
        self.ID, self.OutputPath = self.MakeOutputDir(FolderName=Output)
        self.DailyData = self.NightByNight()
        self.NumberNights = len(self.DailyData)

    def NightByNight(self):
        '''
        Slice the data into night by night elements.
        '''
        #Find where the location

        Time = self.ParamValues[:,0]
        Locations = np.where(np.diff(Time)>0.20)[0]
        SlicedData = []
        Start = 0
        for ChunkCount in range(len(Locations)+1):
            if ChunkCount<len(Locations):
                Stop = Locations[ChunkCount]
            else:
                Stop = len(Time)
            SlicedData.append(self.ParamValues[Start:Stop])
            Start = Stop
        return SlicedData


    def MakeOutputDir(self, FolderName=""):
        '''
        Check if the output directory exists
        '''

        if FolderName:
            ID = FolderName
        else:
            ID = hex(int(time()*1000)).replace('0x16',"")
        OutputDir = os.getcwd()+"/Output"

        if not(os.path.exists(OutputDir)):
            os.system("mkdir %s" %OutputDir.replace(" ", "\ "))

        ResultDir = OutputDir+"/"+ID

        if not(os.path.exists(ResultDir)):
            print("Creating the folder.")
            os.system("mkdir %s" %ResultDir.replace(" ", "\ "))
        else:
            print("The output folder already exists. Deleting all previous files and folders within it.")
            os.system("rm  -rf %s/*" %ResultDir.replace(" ", "\ "))
        return ID, ResultDir



    @property
    def MaskFlares(self):
        '''
        Check if the output directory exists
        '''
        input("Inside masking flares. Yet to be implemented")
        pass



    def NormalizeFlux(self):
        '''
        This method normalizes the flux
        '''
        pass



    def GetNormalizedFlux(self):
        '''
        This method normalizes the flux
        '''
        input("Normalized Flux")
        pass


    def GetDataByNight(NightNum):
        '''
        NightNumber is the night with the first case being 1.
        '''
        input("Inside get data by night")
        pass


    def GetTimeFluxByNight(NightNum):
        '''
        NightNumber is the night with the first case being 2.
        '''
        input("Inside get Time Flux  by night")
        pass
