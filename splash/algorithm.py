'''
This file containts the run algorithm for the transit search
'''

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colors
import itertools
import numpy as np
import multiprocessing as mp
import os
import glob
from tqdm import tqdm
import pickle
from transitleastsquares import transitleastsquares
import splash
from astropy.time import Time

from scipy.stats import binned_statistic, chi2
from math import ceil

from lmfit import Minimizer, Parameters
from lmfit.printfuncs import report_fit

import batman

from .target import Target
from .Functions import ParseFile, SVDSolver, \
      TransitBoxModel, FindLocalMaxima, RunningResidual


class GeneralTransitSearch:
    '''
    Description
    ------------
    This method implements the linearized search for the transits


    Parameters
    ---------------
    This method does a night by night basis for the function

    Current default method is SVD
    method="svd"

    More methods to be implemented in the future.

    Linearize the flux as a function of the
    Yields
    ---------------

    '''

    def __init__(self):
        '''
        Initialized the transit search parameters from Search.ini
        '''

        Filepath = os.path.join(os.getcwd(),"SearchParams.config")

        if os.path.exists(Filepath):
            print("Using user defined search parameters.")
            self.transitSearchParam = ParseFile(Filepath)
        else:
            print("Reading from default search parameters.")
            self.transitSearchParam = ParseFile(os.path.join(splash.__path__,"SearchParams.config"))

    def Run(self, Target, ShowPlot=False, SavePlot=False, SaveData=True, SaveAmplifiedLC=False):
        '''
        Runs SVD algorithm to find the

        Parameters
        -------------

        Target: class object from splash
                SPECULOOS target whose method allows access to the data

        ShowPlot: bool
                  Shows the plot if True

        SavePlot: bool
                  Save the plot if True

        SaveData: bool
                  Save the pickled data of pickled data

        SaveAmplifiedLC: bool
                  Save Figures and data for the amplified light curves
        '''

        self.TStepSize = float(self.transitSearchParam["TStepSize"])/(24.0*60.0)
        self.TDurStepSize = float(self.transitSearchParam["TDurStepSize"])/(24.0*60.0)
        self.TDurLower, self.TDurHigher = [float(Item)/(24.0*60.0) for Item in self.transitSearchParam["TransitDuration"].split(",")]
        self.TDurValues = np.arange(self.TDurLower, self.TDurHigher+self.TDurStepSize, self.TDurStepSize)
        self.BasisItems = self.transitSearchParam['Params'].split(",")
        self.TLS_Status = False
        self.TransitMatch_Status = False
        self.SaveAmplifiedLC = SaveAmplifiedLC

        if self.SaveAmplifiedLC:
            self.AmplifiedFigLoc = "%s/AmplifiedLCs/Figures" %Target.ResultDir
            self.AmplifiedDataLoc = "%s/AmplifiedLCs/Data" %Target.ResultDir
            os.system("mkdir %s/AmplifiedLCs" %Target.ResultDir)
            os.system("mkdir %s" %self.AmplifiedFigLoc)
            os.system("mkdir %s" %self.AmplifiedDataLoc)


        NCPUs = int(self.transitSearchParam["NCPUs"])

        if NCPUs==-1:
            self.NUM_CORES = mp.cpu_count()
        elif NCPUs>0 and NCPUs<65:
            self.NUM_CORES = int(NCPUs)
        else:
            raise("Cannot have more than 64 cores...")

        self.ParamColumns = self.getParamCombination(Target)
        self.AllMetricMatrix = []
        self.AllTransitDepthMatrix = []
        self.AllTransitDepthUnctyMatrix = []
        self.AllResidualMatrix = []
        self.AllCombinationBasis = []
        self.AllSTD = []
        self.AllCoeffValues = []

        self.AllModeledT0 = []
        self.AllDetrendedFlux = []

        for NightNum in range(Target.NumberOfNights):
            print("Running %d Night" %(NightNum+1))

            RunStatus = self.RunNight(Target, NightNum)

            if RunStatus==0:
                    self.AllCombinationBasis.append([])
                    #Save the data
                    self.AllMetricMatrix.append(np.array([]))
                    self.AllTransitDepthMatrix.append(np.array([]))
                    self.AllTransitDepthUnctyMatrix.append(np.array([]))
                    self.AllResidualMatrix.append(np.array([]))
                    self.AllCoeffValues.append([])
                    self.AllSTD.append(np.std(Target))

            elif RunStatus==1:
                self.AllCombinationBasis.append(self.BestBasisColumns)
                self.AllDetrendedFlux.extend(self.BestDetrendedModel)

                #Save the data
                self.AllMetricMatrix.append(self.CurrentMetricMatrix)
                self.AllTransitDepthMatrix.append(self.CurrentTransitDepthMatrix)
                self.AllTransitDepthUnctyMatrix.append(self.CurrentUnctyTransitMatrix)
                self.AllResidualMatrix.append(self.CurrentResidualMatrix)

                #Now save the data as a pickle
                self.DataDir = os.path.join(Target.ResultDir, "Data")

                if not(os.path.exists(self.DataDir)):
                    os.system("mkdir %s" %self.DataDir)

                if SaveData:
                    FileName = os.path.join(self.DataDir, "Night%sData.pkl" %(NightNum+1))

                    with open(FileName, 'wb') as f:
                        pickle.dump(self.CurrentTransitDepthMatrix, f)
                        pickle.dump(self.CurrentUnctyTransitMatrix, f)
                        pickle.dump(self.CurrentResidualMatrix, f)


                Title = Target.ParamNames[self.BestBasisColumns]
                TitleText = "  ".join(Title)+"\n"


                if ShowPlot or SavePlot:

                    BestPixelRow, BestPixelCol = np.where(self.CurrentMetricMatrix==np.max(self.CurrentMetricMatrix))
                    BestT0 = self.T0_Values[BestPixelRow][0]
                    BestTDur = self.TDurValues[BestPixelCol][0]

                    TitleText += "T0: "+str(round(BestT0,5))+"\n"
                    TitleText += "TDur: "+str(round(BestTDur,1))+"\n"
                    TitleText += Time(2450000+BestT0,format='jd').isot[:10]

                    #Bin the data
                    NumBins = int((max(self.CurrentTime) - min(self.CurrentTime))*24.0*60.0/10.0)
                    NData  = len(self.CurrentTime)/NumBins
                    self.CurrentResidual = self.CurrentFlux - self.BestModel
                    self.BinnedTime = binned_statistic(self.CurrentTime, self.CurrentTime, bins=NumBins)[0]
                    self.BinnedFlux = binned_statistic(self.CurrentTime, self.CurrentFlux, bins=NumBins)[0]

                    #Find the running errorbar
                    self.BinnedResidual = RunningResidual(self.CurrentTime, self.CurrentFlux-self.BestModel, NumBins)

                    T0_Int = int(min(self.T0_Values))

                    XPlot = self.CurrentTime - T0_Int
                    YPlot = self.CurrentFlux
                    T0Offset = np.mean(np.diff(self.T0_Values))/2.0
                    TDurOffSet = np.mean(np.diff(self.TDurValues))/2

                    fig= plt.figure(figsize=(20,14))
                    spec = gridspec.GridSpec(nrows=20, ncols=20, figure=fig)
                    spec.update(hspace=0.025)
                    spec.update(wspace=0.0)

                    ax0 = fig.add_subplot(spec[0:8, 0:18])
                    ax1 = fig.add_subplot(spec[8:11, 0:18])
                    ax2 = fig.add_subplot(spec[11:20, 0:18])
                    ax3 = fig.add_subplot(spec[11:20, 18:20])

                    ax0.plot(self.CurrentTime - T0_Int, self.CurrentFlux,\
                               linestyle="None", color="cyan", marker="o", \
                               markersize=4.5)

                    ax0.errorbar(self.BinnedTime - T0_Int, self.BinnedFlux, \
                                   yerr=self.BinnedResidual,\
                                   marker="o", markersize=7, linestyle="None", \
                                   capsize=3, elinewidth=2, \
                                   color="black", ecolor="black")

                    MedianFlux = np.median(self.CurrentFlux)
                    ax0.plot(self.CurrentTime - T0_Int, self.BestModel, "g-", lw=2)
                    ax0.plot(self.CurrentTime - T0_Int, MedianFlux+self.BestModel-self.DetrendedModel, \
                    "r-", lw=5.0, label="Model")

                    ax0.axvline(BestT0 - T0_Int, color="red")
                    ax0.set_xlim(min(self.T0_Values- T0_Int-T0Offset/2), max(self.T0_Values- T0_Int)+T0Offset/2)

                    YLower, YUpper = np.percentile(self.CurrentFlux, [2.0, 98.0])
                    ax0.set_ylim([YLower, YUpper])
                    ax0.set_xticklabels([])
                    ax0.set_ylabel("Normalized Flux", fontsize=20)
                    ax0.set_title(TitleText)

                    MaxLikelihoodValues = np.max(self.CurrentMetricMatrix,axis=1)
                    PeakLocations = np.where(FindLocalMaxima(MaxLikelihoodValues, NData=4))[0]

                    ax1.axvline(x=BestT0,color="red", lw=2.0, linestyle="-")

                    for Location in PeakLocations:
                        ax1.plot(self.T0_Values[Location], MaxLikelihoodValues[Location], color="red", \
                        marker="d", markersize=10, zorder=20)
                        #ax1.axvline(x=self.T0_Va``lues[Location],color="blue", lw=2.5, linestyle=":")

                    ax1.plot(self.T0_Values, MaxLikelihoodValues, color="black", marker="2", markersize=15, \
                    linestyle=":", lw=0.5)
                    ax1.set_ylim([min([0.6*min(MaxLikelihoodValues), 1.4*min(MaxLikelihoodValues)]), 1.4*max(MaxLikelihoodValues)])
                    ax1.set_yticks([])
                    ax1.set_ylabel("Likelihood", fontsize=15)
                    ax1.set_xticks([])
                    ax1.set_xlim(min(self.T0_Values-T0Offset/2), max(self.T0_Values)+T0Offset/2)


                    ax2.set_ylabel("Transit Duration (mins)", fontsize=20)
                    ax2.imshow(self.CurrentMetricMatrix.T, aspect='auto', origin='lower', \
                          extent=[min(self.T0_Values- T0_Int-T0Offset), max(self.T0_Values - T0_Int+T0Offset), \
                          min(self.TDurValues-TDurOffSet)*24.0*60.0, \
                          max(self.TDurValues+TDurOffSet)*24.0*60.0],
                          norm=colors.PowerNorm(gamma=1./2.0))

                    ax2.axvline(BestT0 - T0_Int, color="black", linestyle=":", lw=1.5)
                    ax2.axhline(BestTDur*24.0*60.0, color="black", linestyle=":", lw=1.5)
                    ax2.set_ylabel("Transit Duration (mins)", fontsize=20)
                    ax2.set_xlabel("Time %s JD " %(T0_Int), fontsize=20)

                    HighestColumn = np.where(np.max(MaxLikelihoodValues)==MaxLikelihoodValues)[0][0]
                    TraDurColumn = self.CurrentMetricMatrix[HighestColumn,:]
                    Index  = np.arange(len(TraDurColumn))
                    ax3.plot(TraDurColumn, Index,"r-", lw=2.5)
                    ax3.set_xticks([])
                    ax3.set_yticks([])

                    if SavePlot:
                        self.DailyBestFolder = os.path.join(Target.ResultDir, "DailyBestCases")
                        if not(os.path.exists(self.DailyBestFolder)):
                            os.system("mkdir %s" %self.DailyBestFolder)
                        SaveName = os.path.join(self.DailyBestFolder,"Night"+str(NightNum+1).zfill(4)+".png")
                        plt.savefig(SaveName)
                    if ShowPlot:
                        plt.show()
                    plt.close('all')

                if SaveAmplifiedLC:

                    AmplifiedLCList = np.array(glob.glob(self.AmplifiedDataLoc+"/*.txt"))
                    AmplifiedT0List = np.array([float(Item.split("/")[-1][:10]) for Item in AmplifiedLCList])

                    BestPixelRow, BestPixelCol = np.where(self.CurrentMetricMatrix==np.max(self.CurrentMetricMatrix))
                    BestT0 = self.T0_Values[BestPixelRow][0]
                    BestTDur = self.TDurValues[BestPixelCol][0]

                    #Bin the data
                    NumBins = int((max(self.CurrentTime) - min(self.CurrentTime))*24.0*60.0/10.0)
                    NData  = len(self.CurrentTime)/NumBins
                    self.CurrentResidual = self.CurrentFlux - self.BestModel
                    self.BinnedTime = binned_statistic(self.CurrentTime, self.CurrentTime, bins=NumBins)[0]
                    self.BinnedFlux = binned_statistic(self.CurrentTime, self.CurrentFlux, bins=NumBins)[0]

                    #Find the running errorbar
                    self.BinnedResidual = RunningResidual(self.CurrentTime, self.CurrentFlux-self.BestModel, NumBins)

                    T0_Int = int(min(self.T0_Values))

                    XPlot = self.CurrentTime - T0_Int
                    YPlot = self.CurrentFlux
                    T0Offset = np.mean(np.diff(self.T0_Values))/2.0
                    TDurOffSet = np.mean(np.diff(self.TDurValues))/2

                    MaxLikelihoodValues = np.max(self.CurrentMetricMatrix,axis=1)
                    PeakLocations = np.where(FindLocalMaxima(MaxLikelihoodValues, NData=4))[0]
                    NumPeaks = len(PeakLocations)
                    T0Peaks = self.T0_Values[PeakLocations]


                    Row=20
                    Col = 14+7*NumPeaks
                    GridRow = 25+7*NumPeaks
                    GridCol = 20

                    fig= plt.figure(figsize=(Row,Col))

                    spec = gridspec.GridSpec(nrows=GridRow, ncols=GridCol, figure=fig)
                    spec.update(hspace=0.025)
                    spec.update(wspace=0.0)

                    ax0 = fig.add_subplot(spec[0:8, 0:18])
                    ax1 = fig.add_subplot(spec[8:11, 0:18])
                    ax2 = fig.add_subplot(spec[11:20, 0:18])
                    ax3 = fig.add_subplot(spec[11:20, 18:20])

                    ax0.plot(self.CurrentTime - T0_Int, self.CurrentFlux,\
                               linestyle="None", color="cyan", marker="o", \
                               markersize=4.5)

                    ax0.errorbar(self.BinnedTime - T0_Int, self.BinnedFlux, \
                                   yerr=self.BinnedResidual,\
                                   marker="o", markersize=7, linestyle="None", \
                                   capsize=3, elinewidth=2, \
                                   color="black", ecolor="black")

                    MedianFlux = np.median(self.CurrentFlux)
                    #ax0.plot(self.CurrentTime - T0_Int, self.BestModel, "g-", lw=2)
                    #ax0.plot(self.CurrentTime - T0_Int, MedianFlux+self.BestModel-self.DetrendedModel, \
                    #"r-", lw=5.0, label="Model")

                    ax0.axvline(BestT0 - T0_Int, color="red")
                    ax0.set_xlim(min(self.T0_Values- T0_Int-T0Offset/2), max(self.T0_Values- T0_Int)+T0Offset/2)

                    YLower, YUpper = np.percentile(self.CurrentFlux, [2.0, 98.0])
                    ax0.set_ylim([YLower, YUpper])
                    ax0.set_xticklabels([])
                    ax0.set_ylabel("Normalized Flux", fontsize=20)
                    ax0.set_title(TitleText)


                    ax1.axvline(x=BestT0,color="red", lw=2.0, linestyle="-")

                    for Location in PeakLocations:
                        ax1.plot(self.T0_Values[Location], MaxLikelihoodValues[Location], color="red", \
                        marker="d", markersize=10, zorder=20)
                        #ax1.axvline(x=self.T0_Va``lues[Location],color="blue", lw=2.5, linestyle=":")

                    ax1.plot(self.T0_Values, MaxLikelihoodValues, color="black", marker="2", markersize=15, \
                    linestyle=":", lw=0.5)
                    ax1.set_ylim([min([0.6*min(MaxLikelihoodValues), 1.4*min(MaxLikelihoodValues)]), 1.4*max(MaxLikelihoodValues)])
                    ax1.set_yticks([])
                    ax1.set_ylabel("Likelihood", fontsize=15)
                    ax1.set_xticks([])
                    ax1.set_xlim(min(self.T0_Values-T0Offset/2), max(self.T0_Values)+T0Offset/2)


                    ax2.set_ylabel("Transit Duration (mins)", fontsize=20)
                    ax2.imshow(self.CurrentMetricMatrix.T, aspect='auto', origin='lower', \
                          extent=[min(self.T0_Values- T0_Int-T0Offset), max(self.T0_Values - T0_Int+T0Offset), \
                          min(self.TDurValues-TDurOffSet)*24.0*60.0, \
                          max(self.TDurValues+TDurOffSet)*24.0*60.0],
                          norm=colors.PowerNorm(gamma=1./2.0))

                    ax2.axvline(BestT0 - T0_Int, color="black", linestyle=":", lw=1.5)
                    ax2.axhline(BestTDur*24.0*60.0, color="black", linestyle=":", lw=1.5)
                    ax2.set_ylabel("Transit Duration (mins)", fontsize=20)
                    ax2.set_xlabel("Time %s JD " %(T0_Int), fontsize=20)

                    HighestColumn = np.where(np.max(MaxLikelihoodValues)==MaxLikelihoodValues)[0][0]
                    TraDurColumn = self.CurrentMetricMatrix[HighestColumn,:]
                    Index  = np.arange(len(TraDurColumn))
                    ax3.plot(TraDurColumn, Index,"r-", lw=2.5)
                    ax3.set_xticks([])
                    ax3.set_yticks([])

                    for Counter,CurrentT0Value in enumerate(T0Peaks):
                        StartRow=22+Counter*7
                        StopRow =22+(Counter+1)*7

                        FileIndex = np.argmin(np.abs(AmplifiedT0List-CurrentT0Value))
                        SelectedFile = AmplifiedLCList[FileIndex]

                        ax = fig.add_subplot(spec[StartRow:StopRow, 0:18])

                        FileData = np.loadtxt(SelectedFile, delimiter=",")
                        #FindMe

                        TimeSeries = FileData[:,0] - T0_Int
                        Model = FileData[:,2]
                        DetrendedFlux = FileData[:,3]

                        ax.plot(TimeSeries, self.CurrentFlux,\
                                   linestyle="None", color="cyan", marker="o", \
                                   markersize=4.5)

                        ax.errorbar(self.BinnedTime - T0_Int, self.BinnedFlux, \
                                       yerr=self.BinnedResidual,\
                                       marker="o", markersize=7, linestyle="None", \
                                       capsize=3, elinewidth=2, \
                                       color="black", ecolor="black")


                        ax.axvline(x=CurrentT0Value-T0_Int, color="red", linestyle=":")

                        ax.plot(TimeSeries, Model, "r-")

                        ax.plot(TimeSeries, Model-DetrendedFlux+MedianFlux, "g-")

                        ax.set_xlim(min(self.T0_Values- T0_Int-T0Offset/2), max(self.T0_Values- T0_Int)+T0Offset/2)

                    SaveName = self.AmplifiedFigLoc+"/"+str(NightNum+1).zfill(5)+".png"
                    plt.savefig(SaveName)
                    plt.close('all')




        #Convert lists to array
        self.AllModeledT0 = np.array(self.AllModeledT0)
        self.AllDetrendedFlux = np.array(self.AllDetrendedFlux)

        #Save the file
        np.savetxt(os.path.join(self.DataDir,"DetrendedFlux.csv"), \
            np.transpose((Target.AllTime, self.AllDetrendedFlux)), \
            delimiter="," , header="Time, Detrended Flux")


    def getParamCombination(self, Target):
        '''
        This method will yield different combination of the parameters
        to be used as the basis vector.

        Parameter:
        -----------
        Target: class
                Target class that allows access to the lightcurve


        Yields
        -----------
        The combination of column numbers of data which are to be
        tested as the basis vector.
        '''

        ColumnValues = []

        print("Detrending basis vectors...")
        for Basis in self.BasisItems:
            for ItemCount, Item in enumerate(Target.ParamNames):

                if Basis.upper() in Item.upper():
                    print(Item.upper())
                    ColumnValues.append(ItemCount)

        ColumnValues = list(set(ColumnValues))

        ColumnArray = np.array(ColumnValues)
        self.ColumnArray = np.array(ColumnValues)

        self.BasisCombination = []
        for i in range(1,int(self.transitSearchParam["Combination"])+1):
            Basis = [list(itertools.combinations(self.ColumnArray,i))]
            self.BasisCombination.extend(Basis[0])



    def ConstructBasisMatrix(self, T0, TDur, BasisColumn):
       '''
       This method constructs

       Parameters
       ============
       T0: float
           The value of

       TDur: float
            The value of

       BasisColumn: list of integers
                    The value of columns of CurrentData to be used to used as basis functions


       Yields
       ==============
       Basis vector which can be used to
       '''

       PolyOrder = int(self.transitSearchParam['PolynomialOrder'])
       NumParams = PolyOrder*len(BasisColumn)+2
       BasisMatrix = np.ones((len(self.CurrentTime), NumParams))

       for Col_Count, Col in enumerate(BasisColumn):
           for Order in range(PolyOrder):
               AssignColumn = Col_Count*PolyOrder+Order
               BasisMatrix[:,AssignColumn] = np.power(self.CurrentData[:, Col],Order+1)

       BasisMatrix[:,-2] =  TransitBoxModel(self.CurrentTime, T0, TDur)
       return BasisMatrix


    def RunNight(self, Target, NightNum):
        '''
        This functions runs SVD to find transit like feature on a target light
        curve for each night.


        Parameter:
        -----------
        Target: class
                Target class that allows access to the lightcurve


        NightNum: integer
                  The number night is the index of the night to be run e.g. 1
                  Indexing begins at 1.

        Yields
        ==============
        Yields 2D chi squared map (M,N) for each night for the
        where M is the length of T0 Values and N is the length
        of Transit duration arrays. Can be accessed using ChiSquareMap.

        Returns:
        -------------
        0 if not enough data
        1 if successful

        '''


        self.CurrentData = Target.DailyData[NightNum]
        self.CurrentTime = self.CurrentData[:,0]
        self.CurrentFlux = self.CurrentData[:,1]

        #If not enough data points return 0
        if len(self.CurrentTime)<20:
            return 0

        Offset = 5./(60.0*24.0)
        self.T0_Values = np.arange(self.CurrentTime[0]+Offset,self.CurrentTime[-1]-Offset, self.TStepSize)

        NewT0Values = []
        #Remove the data which have more than 5 minutes
        for Value in self.T0_Values:
            MinDifference = np.min(np.abs(self.CurrentTime-Value))

            if MinDifference<Offset:
                NewT0Values.append(Value)

        self.T0_Values = np.array(NewT0Values)
        self.CurrentMetricMatrix = np.ones((len(self.T0_Values), len(self.TDurValues)))*-np.inf
        self.CurrentResidualMatrix = np.ones((len(self.T0_Values), len(self.TDurValues)))*-np.inf
        self.CurrentTransitDepthMatrix = np.ones((len(self.T0_Values), len(self.TDurValues)))*-np.inf
        self.CurrentUnctyTransitMatrix = np.ones((len(self.T0_Values), len(self.TDurValues)))*-np.inf
        self.BestMetric = -np.inf

        self.AllModeledT0.extend(self.T0_Values)

        #Need to run for all of the cases...
        T0_TDur_Basis_Combinations = itertools.product(self.T0_Values, self.TDurValues, self.BasisCombination)
        NumOperations = (len(self.T0_Values)*len(self.TDurValues)*len(self.BasisCombination))

        #If no data for the night
        if len(self.CurrentData)<1:
            print("For Night:", NightNum+1, " no data is available")
            return 0

        for i in tqdm(range(ceil(NumOperations/self.NUM_CORES))):
            Tasks = []
            CPU_Pool = mp.Pool(self.NUM_CORES)
            for TaskCounter in range(self.NUM_CORES):
                try:
                    T0, TDur, Combination = next(T0_TDur_Basis_Combinations)
                except:
                    pass
                BasisVector = self.ConstructBasisMatrix(T0, TDur, Combination)
                Tasks.append(CPU_Pool.apply_async(SVDSolver,(BasisVector, self.CurrentFlux, T0, TDur, Combination)))

            CPU_Pool.close()
            CPU_Pool.join()


            for Index, Task in enumerate(Tasks):
                Coeff, Uncertainty, Residual, Model, DetrendedFlux, \
                T0, TDur, Combination = list(Task.get())
                T0_Index = np.argmin(np.abs(self.T0_Values-T0))
                TDur_Index = np.argmin(np.abs(self.TDurValues-TDur))

                #Metric to determine which is the best target from the night
                #Metric = (Coeff[-2]/Uncertainty[-2])
                #Metric = (Coeff[-2]/Uncertainty[-2])/(Residual)
                Metric = (Coeff[-2]/Uncertainty[-2])/(Residual*Residual)

                if self.BestMetric<Metric:
                    self.BestCoeff = Coeff
                    self.BestMetric = Metric
                    self.BestBasisColumns = np.array(Combination)
                    self.BestModel = Model
                    self.DetrendedModel = DetrendedFlux
                    self.BestDetrendedModel = self.CurrentFlux - DetrendedFlux
                    self.CurrentSTD = np.std(self.CurrentFlux - Model)


                if self.CurrentMetricMatrix[T0_Index, TDur_Index]<Metric:
                    T0BestMetric = np.max(self.CurrentMetricMatrix[T0_Index,:])<Metric
                    self.CurrentResidualMatrix[T0_Index, TDur_Index] = Residual
                    self.CurrentTransitDepthMatrix[T0_Index, TDur_Index] = Coeff[-2]
                    self.CurrentUnctyTransitMatrix[T0_Index, TDur_Index] = Uncertainty[-2]
                    self.CurrentMetricMatrix[T0_Index, TDur_Index] = Metric

                    if self.SaveAmplifiedLC and T0BestMetric:
                        self.CurrentAmpSaveName = os.path.join(self.AmplifiedDataLoc,str("%.5f" %T0))+".txt"
                        np.savetxt(self.CurrentAmpSaveName, np.transpose((self.CurrentTime, self.CurrentFlux, Model, DetrendedFlux)), delimiter=",",header="Time, Flux, Model, DetrendedFlux")
        #Replace the missed value with minima
        #InfLocation = np.where(~np.isfinite(self.CurrentMetricMatrix))
        InfLocation = np.where(np.logical_or(~np.isfinite(self.CurrentMetricMatrix),np.abs(self.CurrentTransitDepthMatrix)>0.3))

        MedianResidual = np.median(self.CurrentResidualMatrix)
        #STD = np.std(self.BestDetrendedModel)
        self.CurrentResidualMatrix[InfLocation] = np.max(self.CurrentResidualMatrix)
        self.CurrentTransitDepthMatrix[InfLocation] = 1e-8
        self.CurrentUnctyTransitMatrix[InfLocation] = 1.0
        self.CurrentMetricMatrix[InfLocation] = 0.0

        self.CurrentMetricMatrix*=self.CurrentSTD*self.CurrentSTD
        self.AllSTD.append(self.CurrentSTD)
        self.AllCoeffValues.append(self.BestCoeff)
        return 1



    def PeriodicSearch(self, Target, MinPeriod=0.5, method="TransitMatch", SavePlot=True, ShowPlot=False):
        '''
        This function utilizes the linear search information and

        Parameters
        ------------

        Target: splash target object
                Target object initiated with a light curve file

        MinPeriod: float
                Minimum period to search for transit

        method: string
                either "TransitMatch" or "TLS" is expected. Transit pair algorithm run

        ShowPlot: bool
                Shows the plot if True

        SavePlot: bool
                 Saves the plot if True under DiagnosticPlots subfolder


        Yields
        ---------------
        '''

        #The minimum period for which to look the planet
        self.MinPeriod = MinPeriod

        self.UnravelMetric = []
        Row, Col = np.shape(self.AllMetricMatrix[0])
        self.AllArrayMetric = np.zeros((Col,1))
        self.TransitDepthArray = np.zeros((Col,1))
        self.TransitUnctyArray = np.zeros((Col,1))
        self.ResidualArray = np.zeros((Col,1))

        for Counter in range(len(self.AllMetricMatrix)):
            self.AllArrayMetric= np.column_stack((self.AllArrayMetric, self.AllMetricMatrix[Counter].T))
            self.TransitDepthArray = np.column_stack((self.TransitDepthArray, self.AllTransitDepthMatrix[Counter].T))
            self.TransitUnctyArray = np.column_stack((self.TransitUnctyArray, self.AllTransitDepthUnctyMatrix[Counter].T))
            self.ResidualArray  = np.column_stack((self.ResidualArray , self.AllResidualMatrix[Counter].T))

        self.AllArrayMetric = self.AllArrayMetric[:,1:]
        self.UnravelMetric = np.max(self.AllArrayMetric, axis=0)

        self.TransitDepthArray = self.TransitDepthArray[:,1:]
        self.TransitUnctyArray = self.TransitUnctyArray[:,1:]
        self.ResidualArray = self.ResidualArray[:,1:]


        #Determine the phase coverage
        if method == "TransitMatch":
            self.TransitMatch(Target, MinPeriod, ShowPlot, SavePlot)

        elif method == "TLS":
            print("Now running TLS")
            self.TLS(Target, MinPeriod, ShowPlot, SavePlot)

        elif method == "DoubleAnneal":
            self.DoubleAnneal(Target, MinPeriod, ShowPlot, SavePlot)

        else:
            raise ValueError("No such method was found, Has to be TLS or DoubleAnneal")



    def TransitMatch(self, Target, MinPeriod, ShowPlot, SavePlot):
           '''
           Method to look at periodicity in the likelihood function

           Parameters
           ------------

           Target: splash target object
                   Target object initiated with a light curve file

           MinPeriod: float
                  Minimum period to search for transit

           ShowPlot: bool
                   Shows the plot if True

           SavePlot: bool
                    Saves the plot if True under DiagnosticPlots subfolder


           Return
           -------
           float, float, float
                    Return value of T0, Period, and Likelihood
           '''



           self.PeakLocations = np.where(FindLocalMaxima(self.UnravelMetric, NData=4))[0]

           #Take all transit pairs
           LocationCombination = list(itertools.combinations(self.PeakLocations,2))


           self.T0s = []
           self.TP_Periods = []
           self.TransitDepth = []
           self.SDE = []


           for Loc1,Loc2 in LocationCombination:
               CurrentPeriod = np.abs(self.AllModeledT0[Loc1]-self.AllModeledT0[Loc2])
               HarmonicPeriod = CurrentPeriod
               HarmonicNum=2
               PreviousPoints=0

               while HarmonicPeriod>MinPeriod:
                   PhaseStepSize = 0.5*float(self.transitSearchParam["TStepSize"])/(HarmonicPeriod*60.0*24.0)

                   #Average Error
                   T1 = self.AllModeledT0[Loc1]
                   T2 = self.AllModeledT0[Loc2]

                   CurrentPhase = (self.AllModeledT0-T1+HarmonicPeriod/2.0)%HarmonicPeriod
                   CurrentPhase = CurrentPhase/HarmonicPeriod

                   SelectedColumns = np.abs(CurrentPhase-0.5)<PhaseStepSize

                   #Calculate the numerator
                   Numerator = np.sum(self.AllArrayMetric[:,SelectedColumns], axis=1)

                   #Calculate the denominator
                   NumPoints = np.sum(SelectedColumns)

                   PreviousPoints = NumPoints
                   #Calculate the mean
                   SelectedTransitDepth = self.TransitDepthArray[:,SelectedColumns]



                   UncertaintyTransit = self.TransitUnctyArray[:,SelectedColumns]
                   Weights = 1./UncertaintyTransit
                   TotalWeights = np.sum(Weights, axis=1)
                   WeightedMean = np.sum((SelectedTransitDepth*Weights).T/TotalWeights,axis=0)

                   #TransitError = np.power(SelectedTransitDepth-WeightedMean,2)/UncertaintyTransit
                   TransitError = np.abs(SelectedTransitDepth.T-WeightedMean.T)/UncertaintyTransit.T

                   DeltaDifference = np.sqrt(np.sum(TransitError,axis=0))
                   #DeltaDifference = np.power(np.sum(TransitError,axis=0),0.40)
                   Likelihood = chi2.sf(DeltaDifference, NumPoints-1)



                   CalcValue = Numerator*Likelihood

                   #print("The calculated value is::", CalcValue)
                   self.T0s.append(min([T1,T2]))
                   self.TP_Periods.append(HarmonicPeriod)

                   BestLocation = CalcValue == np.max(CalcValue)
                   BestTransitDepth = WeightedMean[BestLocation]

                   self.SDE.append(np.max(CalcValue))
                   self.TransitDepth.append(BestTransitDepth)
                   HarmonicPeriod=CurrentPeriod/HarmonicNum
                   HarmonicNum+=1.0



           #Look for the Metric Array
           self.TP_Periods = np.array(self.TP_Periods)
           self.SDE = np.array(self.SDE)
           self.SDE /= np.mean(self.SDE)
           self.T0s = np.array(self.T0s)
           BestPeriod = self.TP_Periods[np.argmax(self.SDE)]

           #Arrange the data
           ArrangeIndex = np.argsort(self.TP_Periods)
           self.TP_Periods = self.TP_Periods[ArrangeIndex]
           self.SDE = self.SDE[ArrangeIndex]
           self.T0s = self.T0s[ArrangeIndex]

           SaveName = os.path.join(self.DataDir, "TransitMatchingPeriodogram.csv")

           np.savetxt(SaveName, np.transpose((self.T0s, self.TP_Periods, self.SDE)),\
           delimiter=",",header="Period,SDE")

           #plotting
           fig, ax1 = plt.subplots(figsize=(14,8))
           ax2= ax1.twinx()

           for i in range(0,4):
               if i == 0:
                   ax1.axvline(x=0.5*BestPeriod, color="cyan", linestyle=":", lw=3, alpha=0.8)
               else:
                   ax1.axvline(x=i*BestPeriod, color="cyan", linestyle=":", lw=3, alpha=0.8)
           ax1.plot(self.TP_Periods, self.SDE, "r-", lw=2)
           ax2.plot(Target.PhasePeriod, Target.PhaseCoverage, color="green", alpha=0.8, lw=2.0, label="Phase Coverage")
           ax1.set_xlabel("Period (Days)", fontsize=20)
           ax2.set_ylabel("Phase Coverage (%)", color="green", labelpad=8.0,fontsize=20, rotation=-90)
           ax1.set_ylabel("Signal Detection Efficiency", color="red", fontsize=20)
           MinXLim = min([min(self.TP_Periods), min(Target.PhasePeriod)])
           MaxXLim = min([max(self.TP_Periods), max(Target.PhasePeriod),25.0])

           ax1.set_xlim([MinXLim, MaxXLim])
           ax1.text(0.90*MaxXLim,0.98*max(self.SDE), "Best Period:"+str(round(BestPeriod,5)), horizontalalignment="right")
           ax1.tick_params(which="both", direction="in", width=2.5, length=12)
           ax2.tick_params(which="both", direction="in", colors="green", width=2.5, length=12)
           ax1.spines['left'].set_color('red')
           ax1.spines['right'].set_color('green')
           ax2.spines['right'].set_color('green')
           ax2.spines['left'].set_color('red')
           plt.tight_layout()

           if SavePlot:
               self.SavePath = os.path.join(Target.ResultDir, "Periodogram")
               if not(os.path.exists(self.SavePath)):
                   os.system("mkdir %s" %self.SavePath)
               if SavePlot:
                   plt.savefig(os.path.join(self.SavePath,"TransitMatching.png"))
               if ShowPlot:
                   plt.show()

           plt.close('all')
           self.TransitMatch_Status = True



    def TLS(self, Target, MinPeriod, ShowPlot, SavePlot):
        '''
        Performs transit least squares search
        on the detrended light curve that preserves the transit

        Parameters
        ------------

        Target: splash target object
                Target object initiated with a light curve file

        MinPeriod: float
                Minimum period to search for transitNumParam="11",

        ShowPlot: bool
                Shows the plot if True

        SavePlot: bool
                 Saves the plot if True under DiagnosticPlots subfolder
        '''

        if not(ShowPlot or SavePlot):
            print("Either SavePlot or SavePlot should be True. Toggling on SavePlot.")
            SavePlot = True

        model = transitleastsquares(Target.AllTime, self.AllDetrendedFlux+1.0)

        results = model.power(
        period_min=MinPeriod,
        oversampling_factor=15,
        duration_grid_step=1.02,
        n_transits_min=2,
        Mstar=0.15,
        Rstar=0.15
        )

        fig, ax = plt.subplots(figsize=(20,14), ncols=1, nrows=2)
        ax2= ax[0].twinx()
        ax[0].axvline(results.period, alpha=0.4, lw=10)
        for n in range(2, 5):
            ax[0].axvline(n*results.period, alpha=0.4, lw=1)
            ax[0].axvline(results.period / n, alpha=0.4, lw=1,)
        ax[0].text(min([25,max(results.periods)*0.90]), max(results.power)*0.95,\
             "Period:"+str(round(results.period,4)), fontsize=20)
        ax[0].set_xlabel('Period (days)', fontsize=20)
        ax[0].plot(results.periods, results.power, color='red', lw=2.0)
        ax[0].set_xlim(min(results.periods), min([max(results.periods),25.0]))
        ax[0].set_ylim([-0.5, max(results.power)*1.05])
        ax2.plot(Target.PhasePeriod, Target.PhaseCoverage, color="green", lw=2)

        ax[0].set_ylabel(r"SDE", color="red", fontsize=20)
        ax2.set_ylabel(r"Phase Coverage", color="green", labelpad=7.0,fontsize=20, rotation=-90)

        ax[0].spines['left'].set_color('red')
        ax[0].spines['right'].set_color('green')
        ax2.spines['right'].set_color('green')
        ax2.spines['left'].set_color('red')

        Residual = results.model_folded_model-results.folded_y

        SelectedDataIndex = np.logical_and(results.folded_phase>0.45,results.folded_phase<0.55)
        SelectedPhase = results.folded_phase[SelectedDataIndex]
        SelectedFlux = results.folded_y[SelectedDataIndex]
        SelectedResidual = Residual[SelectedDataIndex]


        NumBins = int(len(SelectedFlux)/36)

        BinnedPhase = binned_statistic(SelectedPhase, SelectedPhase, statistic='mean', bins=NumBins)[0]
        BinnedFlux = binned_statistic(SelectedPhase, SelectedFlux, statistic='mean', bins=NumBins)[0]
        BinnedResidual = RunningResidual(SelectedPhase, SelectedResidual, NumBins)

        ax[1].plot(results.folded_phase, results.folded_y, color='cyan', \
        marker="o", linestyle="None")
        ax[1].plot(results.model_folded_phase, results.model_folded_model, color='red', lw=2)
        ax[1].errorbar(BinnedPhase, BinnedFlux, yerr=BinnedResidual, capsize=5, \
        marker="o", markersize=5, linestyle="None", color="black")

        ax[1].set_xlim([0.45, 0.55])
        ax[1].set_xlabel('Phase', fontsize=20)
        ax[1].set_ylabel('Relative flux', fontsize=20)

        plt.tight_layout()


        #Now saving the files
        self.SavePath = os.path.join(Target.ResultDir, "Periodogram")
        if not(os.path.exists(self.SavePath)):
            os.system("mkdir %s" %self.SavePath)

        if SavePlot:
            plt.savefig(os.path.join(self.SavePath,"TLS_Periodogram.png"))
        if ShowPlot:
            plt.show()
        plt.close('all')


    def DoubleAnneal(self, Target, MinPeriod, ShowPlot=False, SavePlot=True, DataSetNumber=3):
       '''
       Method to look at periodicity in the likelihood function

       Parameters
       ------------

       Target: splash target object
               Target object initiated with a light curve file

       MinPeriod: float
              Minimum period to search for transit

       ShowPlot: bool
               Shows the plot if True. Default value set to False.

       SavePlot: bool
                Saves the plot if True under DiagnosticPlots subfolder. Default value set to True

       DataSetNumber:integer
                The set of data number set to 3. This the set of nights that is used for fitting in double annealing

       Return
       -------
       float, float, float
                Return value of T0, Period, and Likelihood
       '''

       self.PeakLocations = np.where(FindLocalMaxima(self.UnravelMetric, NData=4))[0]


       #Arrange the Unravelled metric by increasing power
       self.SelectedT0s = self.AllModeledT0[self.PeakLocations]
       self.SelectedLikelihood = self.UnravelMetric[self.PeakLocations]

       self.ArrangeIndex = np.argsort(self.SelectedLikelihood)[::-1]

       self.SelectedT0s = self.SelectedT0s[self.ArrangeIndex]
       self.SelectedLikelihood = self.SelectedLikelihood[self.ArrangeIndex]
       self.PeakLocations = self.PeakLocations[self.ArrangeIndex]

       #Take all transit pairs
       LocationCombination = list(itertools.combinations(self.SelectedT0s,2))

       self.TDur = 2.0/24.0

       self.T0s = []
       self.TP_Periods = []
       self.TransitDepth = []
       self.SDE = []

       self.CurrentCaseNumber = 1
       self.AllCaseNumber = len(LocationCombination)


       while self.CurrentCaseNumber<self.AllCaseNumber:

           #Get the data....
           self.T1, self.T2 = LocationCombination[self.CurrentCaseNumber]
           self.CurrentT0 = min([self.T1, self.T2])
           self.CurrentPeriod = abs(self.T1 -self.T2)

           #Data length = 1.5 hours around transit
           self.SelectDataIndex = np.abs((Target.AllTime -self.CurrentT0 +self.TDur/2.)%self.CurrentPeriod)<self.TDur
           self.SelectDataIndex = np.concatenate((np.array([False]), self.SelectDataIndex))

           self.AllBreakLocation = list(np.where(np.diff(self.SelectDataIndex)>0.1)[0])
           CurrentNumNights = len(self.AllBreakLocation)//2

           #Rearrange AllBreakLocation
           TempAllBreakLocation = []
           for counter in range(CurrentNumNights):
               TempAllBreakLocation.append(self.AllBreakLocation[counter*2:(counter+1)*2])

           DataLength = [Y-X for X,Y in TempAllBreakLocation]

           while len(DataLength)>3:
               PopIndex = np.argmin(DataLength)
               DataLength.pop(PopIndex)
               TempAllBreakLocation.pop(PopIndex)

           self.AllBreakLocation = TempAllBreakLocation

           #redefind the select index
           self.SelectDataIndex = np.zeros(len(Target.AllTime)).astype(bool)
           self.NightNumberIndex = []

           for Start,Stop in TempAllBreakLocation:
               self.SelectDataIndex[Start:Stop]=True
               CurrentT0Value = int(Target.AllTime[Start])
               CurrentNightIndex = np.where(Target.Daily_T0_Values == CurrentT0Value)[0][0]
               self.NightNumberIndex.append(CurrentNightIndex)

           #Define the selected time and selected dataset
           self.SelectedTime = Target.AllTime[self.SelectDataIndex]
           self.SelectedFlux = Target.AllFlux[self.SelectDataIndex]
           self.SelectedData = Target.ParamValues[self.SelectDataIndex]


           #Construct the basis matrix
           NCols = 0
           for NightIndex in self.NightNumberIndex:
               NCols+=len(self.AllCombinationBasis[NightIndex])
           NCols*=2


           #Construct Basis
           self.BasisMatrix = np.zeros((len(self.SelectedTime),NCols))

           StartIndex = 0

           AssignCol = 0
           ModelSelection  = ""

           for CurrentNight in  self.NightNumberIndex:
               StartIndex, StopIndex = self.AllBreakLocation[CurrentNight]
               for Col in self.AllCombinationBasis[CurrentNight]:
                   #MeanValue = np.mean(self.SelectedData[StartIndex:StopIndex,Col])
                   self.BasisMatrix[StartIndex:StopIndex, AssignCol] = self.SelectedData[StartIndex:StopIndex,Col]
                   self.BasisMatrix[StartIndex:StopIndex, AssignCol+1] = np.power(self.SelectedData[StartIndex:StopIndex,Col],2)
                   AssignCol += 2
               ModelSelection+=str(len(self.AllCombinationBasis[CurrentNight]))
               StartIndex = StopIndex

           #perform minimization using lmfit arbitrary number of basis vector

           #Jave to initiate the model
           Ps = self.CurrentPeriod*86400.0  #Period in seconds
           a_Rs = (30000/(6.67e-11*Ps*Ps))**0.3333


           LMparams = Parameters()
           LMparams.add(name='Period', value=self.CurrentPeriod, min=0.95*self.CurrentPeriod, max=1.05*self.CurrentPeriod)
           LMparams.add(name='T0', value=self.CurrentT0, min=self.CurrentT0-0.1, max=self.CurrentT0+0.1)
           LMparams.add(name='a_Rs', value=a_Rs, min=2.0, max=10000.0)
           LMparams.add(name='Rp_Rs', value=0.01, min=0.0001, max=1.0)
           LMparams.add(name='b', value=0.5, min=0, max=1.0)
           LMparams.add(name='u', value=0.5, min=0, max=1.0)

           #This section will depend on the ModelSelection
           for counter,Model in enumerate(ModelSelection):
               if counter==0:
                   LMparams.add(name='a11', value=0.0, min=-1e6, max=1e6)
                   LMparams.add(name='a12', value=0.0, min=-1e6, max=1e6)
                   if "2" in Model:
                       LMparams.add(name='a21', value=0.0, min=-1e6, max=1e6)
                       LMparams.add(name='a22', value=0.0, min=-1e6, max=1e6)
               elif counter==1:
                   LMparams.add(name='b11', value=0.0, min=-1e6, max=1e6)
                   LMparams.add(name='b12', value=0.0, min=-1e6, max=1e6)
                   if "2" in Model:
                         LMparams.add(name='b21', value=0.0, min=-1e6, max=1e6)
                         LMparams.add(name='b22', value=0.0, min=-1e6, max=1e6)
               elif counter==2:
                    LMparams.add(name='c11', value=0.0, min=-1e6, max=1e6)
                    LMparams.add(name='c12', value=0.0, min=-1e6, max=1e6)
                    if "2" in Model:
                          LMparams.add(name='c21', value=0.0, min=-1e6, max=1e6)
                          LMparams.add(name='b22', value=0.0, min=-1e6, max=1e6)



           def Residual(params,Time, Flux, Basis, ModelSection, ListStartStop):
              '''
              Residual function

              Parameters
              ----------

              Time: 1D array
                    array of time vector for fitting

              Flux: 1D array
                    array of flux vector for fitting

              Basis: N,M sized array
                  Used for detreding
              '''


              if ModelSection=='1':
                theta = np.array([params['a11'].value,params['a12'].value])

              elif ModelSection=='2':
                theta = np.array([params['a11'].value,params['a12'].value,\
                                  params['a21'].value,params['a22'].value])

              elif ModelSection=='11':
                theta = np.array([params['a11'].value,params['a12'].value,\
                                  params['b11'].value,params['b12'].value])

              elif ModelSection=='12':
                theta = np.array([params['a11'].value,params['a12'].value,\
                                  params['b11'].value,params['b12'].value,\
                                  params['b11'].value,params['b12'].value])

              elif ModelSection=='21':
                theta = np.array([params['a11'].value,params['a12'].value,\
                                  params['a21'].value,params['a22'].value,\
                                  params['b11'].value,params['b12'].value])

              elif ModelSection=='122':
                theta = np.array([params['a11'].value,params['a12'].value,\
                                  params['b11'].value,params['b12'].value,\
                                  params['b21'].value,params['b22'].value,\
                                  params['c11'].value,params['c12'].value,\
                                  params['c21'].value,params['c22'].value])

              elif ModelSection=='212':
                theta = np.array([params['a11'].value,params['a12'].value,\
                                  params['a21'].value,params['a22'].value,\
                                  params['b11'].value,params['b12'].value,\
                                  params['c11'].value,params['c12'].value,\
                                  params['c21'].value,params['c22'].value])

              elif ModelSection=='221':
                print("Case 221")
                theta = np.array([params['a11'].value,params['a12'].value,\
                                  params['a21'].value,params['a22'].value,\
                                  params['b11'].value,params['b12'].value,\
                                  params['b21'].value,params['b22'].value,\
                                  params['c11'].value,params['c12'].value])

              elif ModelSection=='222':
                  theta = np.array([params['a11'].value,params['a12'].value,\
                                    params['a21'].value,params['a22'].value,\
                                    params['b11'].value,params['b12'].value,\
                                    params['b21'].value,params['b22'].value,\
                                    params['c11'].value,params['c12'].value,\
                                    params['c21'].value,params['c22'].value])

              #calculate the inclination
              Inc = np.rad2deg(np.arccos(LMparams['b']/LMparams['a_Rs']))

              print("The value of inclination is::", Inc)
              print("The limb darkening parameter is::", params['u'].value)

              #Initiate the batman fitting parameter
              BatmanParam = batman.TransitParams()
              BatmanParam.t0 = params['T0'].value                        #time of inferior conjunction
              BatmanParam.per = params['Period'].value                   #orbital period
              BatmanParam.rp = params['Rp_Rs'].value                     #planet radius (in units of stellar radii)
              BatmanParam.a = params['a_Rs'].value                       #semi-major axis (in units of stellar radii)
              BatmanParam.inc = Inc                                      #orbital inclination (in degrees)
              BatmanParam.ecc = 0.                                       #eccentricity
              BatmanParam.w = 90.                                        #longitude of periastron (in degrees)
              BatmanParam.u = [params['u'].value]                        #limb darkening coefficients [u]
              BatmanParam.limb_dark = "linear"


              #Model the Transit
              m = batman.TransitModel(BatmanParam, Time)             #initializes model
              TransitModelFlux = m.light_curve(BatmanParam)          #calculates light curve

              TrendLine = np.dot(Basis,theta)
              ModelFlux = TransitModelFlux+TrendLine


              #Correct for the offset
              #for StartIndex,StopIndex in ListStartStop:
              #     Offset =  np.mean(self.SelectedFlux[StartIndex:StopIndex] - ModelFlux[StartIndex:StopIndex])
              #     ModelFlux[StartIndex:StopIndex]+=Offset


              Residual = (self.SelectedFlux - ModelFlux)
              return Residual


           ##################################################################

           DiffLoc = list(np.where(np.diff(self.SelectedTime)>0.1)[0])
           DiffLoc.insert(0,0)
           DiffLoc.append(len(self.SelectedTime))

           ListStartStop = []
           for counter in range(len(DiffLoc)-1):
               ListStartStop.append([DiffLoc[counter], DiffLoc[counter+1]])

           print("The model selection is given by::", ModelSelection)

           UniqueFitName = Minimizer(Residual, LMparams, fcn_args=(self.SelectedTime, self.SelectedFlux, self.BasisMatrix, ModelSelection, ListStartStop))
           UniqueFitName.basinhopping()

           report_fit(UniqueFitName)
           print("use basin hopping...")

           input("This is not yet finalized...")


           #Use lmfit method


           self.AllCaseNumber+=1
