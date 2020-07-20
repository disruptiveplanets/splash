'''
Will contain the core functionality of reading
the data from the Cambridge Pipeline.
'''

import os
import re
from time import time
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.signal import medfilt

from .Functions import ReadTxtData, ReadAllNewFitsData,\
     ReadAllOldFitsData, ParseFile, GetID, FindQuality,\
     moving_average
from emcee import EnsembleSampler
from astropy.timeseries import LombScargle

#formatting for the image
import matplotlib as mpl
#mpl.rc('font',**{'sans-serif':['Helvetica'], 'size':15,'weight':'bold'})
mpl.rc('font',**{'serif':['Helvetica'], 'size':15})
mpl.rc('axes',**{'labelweight':'bold', 'linewidth':1.5})
mpl.rc('ytick',**{'major.pad':22, 'color':'k'})
mpl.rc('xtick',**{'major.pad':10,})
mpl.rc('mathtext',**{'default':'regular','fontset':'cm','bf':'monospace:bold'})
mpl.rc('text.latex',preamble=r'\usepackage{cmbright},\usepackage{relsize},'+r'\usepackage{upgreek}, \usepackage{amsmath}')
mpl.rc('contour', **{'negative_linestyle':'solid'})


class Target:
    '''
    This is a class for a target of data
    '''

    def __init__(self, Location="data", Name ="", Output="", version=1):
        '''
        Expect a continuous data concatenated data with header in the first row.
        First row is expected to be time. Second row is expected to be flux.


        Parameters
        ----------
        Location: string
                  path of the

        Name: string


        Output: string


        Attributes
        ----------
        ParamName: The name of the parameters.
        ParamValue: The values of the parameters.
        '''

        if len(Location)>0 and len(Name)>0:
            if version==0:
                print("Loading from txt file.")
                self.ParamNames, self.ParamValues = ReadTxtData(Location, Name)
            elif version==1:
                print("Loading from fits file.")
                self.ParamNames, self.ParamValues = ReadAllOldFitsData(Location, Name)
            elif version==2:
                input("Still awaiting completion of data processing by Cambridge")
                self.ParamNames, self.ParamValues = ReadAllNewFitsData(Location, Name)
            else:
                raise("Only three version are available.")

        #TBJD offset change
        if min(self.ParamValues[:,0])>2450000.0:
            self.ParamValues[:,0]-=2450000.0
        self.ParamValues[:,1]/=np.nanmedian(self.ParamValues[:,1])

        self.AllTime = self.ParamValues[:,0]
        self.AllFlux = self.ParamValues[:,1]

        #Calculate the number of observation hours
        Diff = np.diff(self.AllTime)
        SelectTObsIndex = Diff<0.01
        self.TotalObservationHours = round(np.sum(Diff[SelectTObsIndex])*24.0,2)

        #Find the id from the database
        self.Name = Name
        if "Sp" in Name:
            self.SpeculoosID = Name
            try:
                self.GaiaID  = GetID(Name, IdType="SPECULOOS")
            except:
                print("Error finding the GAIA ID.")
        elif re.search("[0-9]{17}",Name):
            #GAIA ID is at least 17 digit long
            self.GaiaID = Name
            self.SpeculoosID = GetID(Name, IdType="GAIA")
        else:
            self.SpeculoosID="Not Found"
            self.GaiaID = "Not Found"

        Output = self.SpeculoosID if not(Output) else Output
        #Generate the output folder
        self.MakeOutputDir(FolderName=Output)
        self.OutlierLocation = os.path.join(self.ResultDir, "Outliers")

        self.DailyData = self.NightByNight()
        self.NumberOfNights = len(self.DailyData)


        self.Daily_T0_Values = []
        for counter in range(self.NumberOfNights):
            self.Daily_T0_Values.append(int(min(self.DailyData[counter][:,0])))

        self.Daily_T0_Values = np.array(self.Daily_T0_Values)
        #Flag to produce light curves once the data been processed
        self.QualityFactor = np.ones(len(self.ParamValues[:,0])).astype(np.bool)
        self.PreCleaned = False

        #Break the quality factory into nights
        self.BreakQualityFactor()
        self.PhaseCoverage()
        self.DailyVariance, self.VarianceSeries = self.EstimateVariance()




    def EstimateVariance(self):
        '''
        This method find the standard deviation for each night

        Returns
        -------
        array, array
            array of standard deviation value for a night
            array of standard deviation for a data point
        '''
        DailyVariance = []
        VarianceSeries = []
        for i in range(self.NumberOfNights):
            CurrentData = self.DailyData[i]
            CurrentTime = CurrentData[:,0]
            CurrentFlux = CurrentData[:,1]
            _, Variance = moving_average(CurrentFlux, sigma=5, NumPoint=75)
            DailyVariance.append(np.median(Variance))
            VarianceSeries.extend(Variance)
        return np.array(DailyVariance), np.array(VarianceSeries)


    def NightByNight(self):
        '''
        Slice the data into night by night elements.
        '''
        #Find where the location

        self.NightLocations = []
        Time = self.ParamValues[:,0]
        BreakLocations = np.where(np.diff(Time)>0.20)[0]+1

        SlicedData = []
        Start = 0
        for ChunkCount in range(len(BreakLocations)+1):
            if ChunkCount<len(BreakLocations):
                Stop = BreakLocations[ChunkCount]
            else:
                Stop = len(Time)+1
            SlicedData.append(np.array(self.ParamValues[Start:Stop]))
            self.NightLocations.append([Start, Stop])
            Start = Stop
        return SlicedData


    def MakeOutputDir(self, FolderName=""):
        '''
        Check if the output directory exists, and
        make the directory if they do not exist

        Parameters
        -----------
        FolderName: string
                    Name of the Output directory to be created

        Yields:

        '''

        if FolderName:
            self.ID = FolderName
        else:
            self.ID = hex(int(time()*1000)).replace('0x16',"")
        self.OutputDir = os.getcwd()+"/Output"

        if not(os.path.exists(self.OutputDir)):
            os.system("mkdir %s" %self.OutputDir.replace(" ", "\ "))

        self.ResultDir = self.OutputDir+"/"+self.ID

        if not(os.path.exists(self.ResultDir)):
            print("Creating the folder.")
            os.system("mkdir %s" %self.ResultDir.replace(" ", "\ "))
        else:
            print("The output folder already exists. Deleting all previous files and folders within it.")
            os.system("rm  -rf %s/*" %self.ResultDir.replace(" ", "\ "))
        return 1




    def PreClean(self, CutOff=7, NIter=2, Columns=-1, MinDataCount=50,
                 ShowPlot=False, SavePlot=False):
        '''
        Pre cleans the data

        Parameters
        -------------

        NumPoint: integer
                  Number of points for generating the gaussian function

        NIter: integer

        Columns:-1/1
                -1 - Consider all column except time to look for outlier
                 1 - Consider only the differential flux to look for the outlier

        MinDataCount: integer
                default value 50. Consider the data for the night if at least
                this number of data is present.

        ShowPlot: bool
                 Plots the figure for viewing

        SavePlot: bool
                 Saves the figure at the location of the output cirector

        Return:
            Initiates the quality factor list which can be used to select data

        '''

        StartIndex = 0
        #Measure a quality factor for a data
        for NightNumber in range(self.NumberOfNights):
            CurrentData = self.DailyData[NightNumber]
            nRow, nCol = np.shape(CurrentData)
            CurrentQuality = np.ones(nRow).astype(np.bool)
            CurrentTime = CurrentData[:,0]
            CurrentFlux = CurrentData[:,1]

            #Use only flux. Override the value of nCol
            if Columns ==1:
                nCol = 2

            for j in range(1,nCol):
                if "TIME" in self.ParamNames[j].upper() or "SKY" in self.ParamNames[j].upper() or "AIRMASS" in self.ParamNames[j].upper():
                    continue
                Data2Consider = CurrentData[:,j]
                if len(Data2Consider)<MinDataCount:
                    CurrentQuality = np.zeros(len(CurrentData)).astype(np.bool)
                else:
                    Quality = FindQuality(CurrentTime, Data2Consider, CutOff=7.5, NIter=1)
                    CurrentQuality = np.logical_and(CurrentQuality, Quality)
                    NumDataPoints = np.sum(~CurrentQuality)

                    if NumDataPoints>10:
                        warning("More than 10 points marked with bad quality flag.")

            StartIndex,StopIndex = self.NightLocations[NightNumber]
            self.QualityFactor[StartIndex:StopIndex] = CurrentQuality

            if SavePlot or ShowPlot:
                T0_Int = int(min(CurrentTime))

                plt.figure(figsize=(10,8))
                plt.plot(CurrentTime[CurrentQuality]-T0_Int, CurrentFlux[CurrentQuality], "ko", label="Good Data")
                plt.plot(CurrentTime[~CurrentQuality]-T0_Int, CurrentFlux[~CurrentQuality], "ro", label="Bad Data")
                plt.xlabel("JD "+str(T0_Int), fontsize=25)
                plt.ylabel("Normalized Flux", fontsize=25)
                plt.legend(loc=1)
                plt.tight_layout()
                if SavePlot:
                    if not(os.path.exists(self.OutlierLocation)):
                        os.system("mkdir %s" %self.OutlierLocation)
                    SaveName = os.path.join(self.OutlierLocation, "Night"+str(NightNumber+1).zfill(4)+".png")
                    plt.savefig(SaveName)
                if ShowPlot:
                    plt.show()
                plt.close('all')


        #Now remove the bade index of the data
        self.AllTime = self.AllTime[self.QualityFactor]
        self.AllFlux = self.AllFlux[self.QualityFactor]
        self.ParamValues = self.ParamValues[self.QualityFactor]


        self.DailyData = self.NightByNight()
        self.NumberOfNights = len(self.DailyData)

        self.Daily_T0_Values = []
        for counter in range(self.NumberOfNights):
            self.Daily_T0_Values.append(int(min(self.DailyData[counter][:,0])))

        self.Daily_T0_Values = np.array(self.Daily_T0_Values)
        self.Day2DayVariance, self.VarianceSeries = self.EstimateVariance()

        #Flag to produce light curves once the data been processed
        self.PreCleaned = True


    def BreakQualityFactor(self):
        '''
        Method to break up the quality indices night by night
        '''

        self.QualityFactorFromNight = []
        for Start, Stop in self.NightLocations:
            self.QualityFactorFromNight.append(self.QualityFactor[Start:Stop])


    def PhaseCoverage(self, StepSize=0.05, PLow=0.30, PUp=25.0, NTransits=2, Tolerance=0.005):
        '''
        This function calculates the phase coverage of the data

        ################################
        Parameters
        -----------
        PLow: float
              Lowest Period to consider

        PUp: float
             Largest Period. Default value le

        StepSize: float
                  the stepsize of the period to be considered

        NTransits: integer
                  The number of transits to be used with

        Tolerance:
                    is to see the phase coverage for different phases
        '''


        #Redefine maximum periodicity to be looked for phase coverage
        ExpPeriodCoverage = (max(self.AllTime)-min(self.AllTime))
        if ExpPeriodCoverage<PUp:
            PUp=ExpPeriodCoverage

        self.PhasePeriod  = np.arange(PLow, PUp, StepSize)
        self.PhaseCoverage = np.ones(len(self.PhasePeriod ))

        for Count, Period in enumerate(self.PhasePeriod):
            Phase = self.AllTime%Period
            Phase = Phase[np.argsort(Phase)]
            Diff = np.diff(Phase)

            self.PhaseCoverage[Count] -= Phase[0]/Period
            self.PhaseCoverage[Count] -= (Period-Phase[-1])/Period

            Locations = np.where(Diff>0.005)[0]
            for Location in Locations:
                PhaseUncovered = Phase[Location+1]-Phase[Location]
                self.PhaseCoverage[Count] -= PhaseUncovered/Period

        self.PhaseCoverage*=100.




    def N_PhaseCoverage(self, N=2, PeriodMin=0.5, PeriodMax=np.nan, PeriodStepSize=0.005):
      '''
      Time series: an array of values

      N: integer
         Number of transit to be observed

      PeriodMin:float
                Minimum period to look for the phase coverage

      PeriodMax: float
                Maximum value of the period to consider for the phase coverage

      PeriodStepSize: float
                    Period size for the period for which the data is incomplete...
      '''
      TimeSeries = self.AllTime
      if np.isnan(PeriodMax):
          PeriodMax = (TimeSeries[-1] - TimeSeries[0])

      AllPeriod = np.arange(PeriodMin, PeriodMax, PeriodStepSize)

      PhaseCoverageArray = np.zeros(len(AllPeriod))



      def CheckOverlap(Item1, Item2):
          '''
          Function that checks if there is overlap between two phase ranges

          Item1: [float, float]
                  Phase range for the first case

          Item2: [float, float]
                 Phase range for the second case
          '''
          L1,U1 = Item1
          L2,U2 = Item2
          if U1<L2:
              return False
          elif U2<L1:
              return False
          else:
              return True


      for PeriodCounter, Period in enumerate(AllPeriod):
          Phase = TimeSeries%Period
          Phase=Phase/Period
          BreakLocations = list(np.where(np.abs(np.diff(Phase))>0.005/Period)[0])
          BreakLocations.append(len(TimeSeries)-1)

          StartIndex = 0
          PhaseCoverage = []
          for Counter,StopIndex in enumerate(BreakLocations):
              StartValue = Phase[StartIndex]
              StopValue = Phase[StopIndex]

              if StartValue<StopValue:
                  PhaseCoverage.append([Phase[StartIndex], Phase[StopIndex]])
              else:
                  PhaseCoverage.append([Phase[StartIndex], 1.0])
                  PhaseCoverage.append([0.0, Phase[StopIndex]])
              StartIndex = StopIndex+1

          PhaseCoveredValues = {}
          for i in range(N):
              PhaseCoveredValues[i]=[[0,0]]

          for LeftOver in PhaseCoverage:
              PhaseIndex = 0
              while PhaseIndex<N and LeftOver:

                  ComparisonValues = PhaseCoveredValues[PhaseIndex]
                  NComponents = len(ComparisonValues)

                  LowerValue,UpperValue=LeftOver
                  for CompCounter, CurrentCompValue in enumerate(ComparisonValues):

                      CompLower, CompUpper = CurrentCompValue
                      if CompCounter==0:
                          LeftOver = [LowerValue,UpperValue]

                      OverlapStatus=CheckOverlap(CurrentCompValue, LeftOver)

                      if OverlapStatus:

                          ReplaceValue = [min([LowerValue, CompLower]),max([UpperValue, CompUpper])]
                          LeftOver = [max([LowerValue, CompLower]),min([UpperValue, CompUpper])]
                          PhaseCoveredValues[PhaseIndex][CompCounter] = ReplaceValue
                          break

                      if not(OverlapStatus) and CompCounter==NComponents-1:
                          #print("In appending branch...")
                          PhaseCoveredValues[PhaseIndex].append(LeftOver)
                          LeftOver = None
                          break


                  PhaseIndex+=1


                  #Arrange in ascending order
                  for counter in range(N):
                     Values = np.array(PhaseCoveredValues[counter])
                     FirstValues = [x for x,y in Values]
                     ArrangeIndex = np.argsort(FirstValues)
                     ArrangeValues = Values[ArrangeIndex]
                     PhaseCoveredValues[counter]=list(ArrangeValues)


                  #Check if overlapping current dictionary
                  for counter in range(N):
                     InnerCounter = 0
                     while InnerCounter<len(PhaseCoveredValues[counter])-1:

                       OverlapStatus = CheckOverlap(PhaseCoveredValues[counter][InnerCounter],PhaseCoveredValues[counter][InnerCounter+1])
                       if OverlapStatus:

                           L1, U1 = PhaseCoveredValues[counter][InnerCounter]
                           L2, U2 = PhaseCoveredValues[counter][InnerCounter+1]
                           TempReplaceValue = np.array([min([L1,L2]),max([U1,U2])])
                           TempLeftOver = [max([L1,L2]),min([U1,U2])]

                           PhaseCoveredValues[counter].pop(InnerCounter)
                           PhaseCoveredValues[counter][InnerCounter] = TempReplaceValue
                           PhaseCoverage.append(TempLeftOver)

                       InnerCounter+=1


          for counter in range(N):
               Values = np.array(PhaseCoveredValues[counter])
               FirstValues = [x for x,y in Values]
               ArrangeIndex = np.argsort(FirstValues)
               ArrangeValues = Values[ArrangeIndex]
               PhaseCoveredValues[counter]=list(ArrangeValues)

          #Arrange in ascending order
          for counter in range(N):
                Values = np.array(PhaseCoveredValues[counter])
                FirstValues = [x for x,y in Values]
                ArrangeIndex = np.argsort(FirstValues)
                ArrangeValues = Values[ArrangeIndex]
                PhaseCoveredValues[counter]=list(ArrangeValues)



          CurrentPhaseCoverageValue = 0
          BreakStatus= False
          while not(BreakStatus):
              LowerValue = -np.inf
              UpperValue = np.inf

              for counter in range(N):
                  LowerValue = max([LowerValue,PhaseCoveredValues[counter][0][0]])
                  UpperValue = min([UpperValue,PhaseCoveredValues[counter][0][1]])
                  if PhaseCoveredValues[counter][0][1]>UpperValue:
                      PhaseCoveredValues[counter][0][0]=UpperValue


              if UpperValue>LowerValue:
                  CurrentPhaseCoverageValue+=UpperValue-LowerValue

              for counter in range(N):
                  if PhaseCoveredValues[counter][0][1]<=UpperValue:
                      PhaseCoveredValues[counter].pop(0)
                  if len(PhaseCoveredValues[counter])==0:
                      BreakStatus = True
          PhaseCoverageArray[PeriodCounter]=CurrentPhaseCoverageValue
      PhaseCoverageArray = medfilt(PhaseCoverageArray,5)*100.0
      return AllPeriod, PhaseCoverageArray



    def PreWhitening(self, NCases=None, SavePlot=True, ShowPlot=False):
        '''
        The prewhitening is able to re-iteratively  fit out any sinusoidal signal.

        Parameters
        ----------

        NCases: integer
                Fit for Number of cases. Default is None

        SavePlot: boolean
                  Saves the Diagnostic Plot if toggled on. Default Value is True

        ShowPlot: boolean
                  Shows the Diagnostic plot if toggled on. Default Value is False

        '''
        TargetTime = self.AllTime
        TargetFlux = self.AllFlux

        if not(os.path.exists(self.OutlierLocation)):
            os.system("mkdir %s" %self.OutlierLocation)


        self.CorrectedFlux = np.copy(TargetFlux)

        self.RemoveCase = 1

        while True:
            Freq, LS_Power = LombScargle(self.AllTime, self.CorrectedFlux).autopower()
            LS_Period = 1./Freq

            #Consider period less between 0.5 hours and days
            SelectIndex = np.logical_and(LS_Period>0.5/24., LS_Period<10.0)

            LS_Period = LS_Period[SelectIndex]
            LS_Power = LS_Power[SelectIndex]

            MaxPowerLocation = np.argmax(LS_Power)
            self.CurrentPeriod = LS_Period[MaxPowerLocation]
            MaxPowerValue = LS_Power[MaxPowerLocation]

            SNR_Power = MaxPowerValue/np.mean(LS_Power)

            #Cases for breaking from the loop
            if not(NCases) and (MaxPowerValue<0.10 or SNR_Power<15.0 or self.RemoveCase>5):
                break

            elif NCases:
                if NCases>=self.RemoveCase:
                    break

            print("Now removing %s prominent signal from flux." %self.RemoveCase)

            def Likelihood(theta):
                '''
                Likelihood for fitting a sinudodal
                '''

                Amp, T0, Period = theta

                if T0<0.0 or T0>self.CurrentPeriod:
                    return -np.inf

                if Period<0.75*self.CurrentPeriod or Period>1.25*self.CurrentPeriod:
                    return -np.inf

                Model  = Amp*np.sin(2.0*np.pi*(self.AllTime-T0)/Period)
                Residual = np.power(self.CorrectedFlux - Model,2)/self.VarianceSeries
                Value = -(0.5*np.sum(Residual))

                return Value

            #subtracting the mean value for each night
            for i in range(self.NumberOfNights):
                StartIndex, StopIndex = self.NightLocations[i]
                self.CorrectedFlux[StartIndex:StopIndex] -= np.mean(self.CorrectedFlux[StartIndex:StopIndex])


            #Now fit for power
            self.BestResidual = np.inf
            nWalkers = 30
            Amp_Init = np.random.uniform(0.01,0.01, nWalkers)
            T0_Init = np.random.uniform(0,self.CurrentPeriod, nWalkers)
            Period_Init = np.random.uniform(0.8*self.CurrentPeriod, 1.2*self.CurrentPeriod, nWalkers)
            StartingGuess = np.column_stack((Amp_Init, T0_Init, Period_Init))

            _, NDim = np.shape(StartingGuess)

            sampler = EnsembleSampler(nWalkers, NDim, Likelihood, args=[], threads=8)
            state = sampler.run_mcmc(StartingGuess, 3000,  progress=True)



            #Find the best value and cancel it
            Probability = sampler.lnprobability
            X,Y = np.where(Probability==np.max(Probability))
            BestTheta = sampler.chain[X[0],Y[0],:]
            BestAmp, BestT0, BestPeriod = BestTheta
            BestModel = BestAmp*np.sin(2.0*np.pi*(self.AllTime-BestT0)/BestPeriod)

            plt.figure(figsize=(14,14))
            plt.subplot(311)
            plt.plot(LS_Period, LS_Power, "ko")
            plt.axvline(x=self.CurrentPeriod)
            plt.xscale("log")
            plt.ylabel("Lombscargle Power")
            plt.xlabel("Period")
            TitleText = "Period:" + str(round(self.CurrentPeriod,5))
            plt.title(TitleText)
            plt.subplot(312)
            plt.plot(self.AllTime, self.CorrectedFlux, "ko", label="Original Flux")
            plt.plot(self.AllTime, BestModel, "r-", label="Best Fit Model")
            plt.xlabel("Time (JD)", fontsize=20)
            plt.ylabel("Normalized Flux", fontsize=20)
            plt.legend()
            plt.subplot(313)
            plt.plot(self.AllTime, self.CorrectedFlux - BestModel, "ko")
            plt.xlabel("Time (JD)", fontsize=20)
            plt.ylabel("Corrected Flux", fontsize=20)
            plt.tight_layout()

            if SavePlot:
                SaveName = "SinusoidalFit_Case"+ str(self.RemoveCase).zfill(3)+".png"
                plt.savefig(os.path.join(self.OutlierLocation, SaveName))
            if ShowPlot:
                plt.show()


            self.CorrectedFlux-= BestModel
            self.ParamValues[:,1]-=BestModel
            self.AllFlux -= BestModel
            self.DailyData = self.NightByNight()


            self.RemoveCase+=1
