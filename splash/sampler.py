'''
Will contain module for the quick transit fit
and robust transit fit
'''

import batman
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import os
from scipy.stats import binned_statistic
from .Functions import fold_data, RunningResidual


class TransitFit:

    def __init__(self, Target, TransitSearch, NumFits=3, TDur=1.5, NRuns=5000, \
                method="MCMC", Tolerance=None, ShowPlot=False, SavePlot=True):
        '''
        This uses the thing from results from the target as well as Transit Search

        Parameters
        -----------
        Target: Target class object
                Object that provides access the the light curve

        TransitSearch: Transit Search Object
                        Object that provides access to the transit search algorithm performed
                        on the light curve

        NumFits: Integer
                Number of transit case to be connsidered

        TDur: float
            points around the transit to fit.


        NRuns: int
                Number of steps for running MCMC

        Tolerance: float
                Period and T0 will be allowed to have values between Period-1.2*Tolerance to Period+1.2

        SavesPlot: boolean
                Saves the plot if True

        ShowPlot: boolean
                Shows the plot if True

        '''
        #Convert from from Hours to Days
        self.TDur=TDur/24.0
        self.Tolerance = 0.025
        self.NRuns = int(NRuns)
        self.ShowPlot = ShowPlot
        self.SavePlot = SavePlot

        #see if the target has temperature Log g available

        self.STD = 0.90*np.mean(TransitSearch.AllSTD)

        self.ConstructFitCases(Target, TransitSearch, TDur, NumFits)
        print("MCMC Run completed")


    def GetNightNumber(self, Target):
        '''
        Get which night data are selected among all the nights available. Stores in the
        the variable AllNightIndex.

        Parameters:
        ------------

        Target: class object
                Target class object from splash in order to check which night the data
                for transit fit are selected from.


        '''

        #Find the number of night
        self.BreakLocation = np.where(np.diff(self.SelectedTime)>0.30)[0]
        self.AllNightIndex = []

        StartIndex = 0
        for counter in range(len(self.BreakLocation)+1):
            if counter == len(self.BreakLocation):
                StopIndex = len(self.SelectedTime)
            else:
                StopIndex = self.BreakLocation[counter]+1
            CurrentNight = int(min(self.SelectedTime[StartIndex:StopIndex]))
            CurrentNightIndex = np.where(Target.Daily_T0_Values == CurrentNight)[0][0]
            self.AllNightIndex.append(CurrentNightIndex)
            StartIndex = StopIndex
        self.AllNightIndex = np.array(self.AllNightIndex)


    def ConstructFitCases(self, Target, TransitSearch, TDur, NumFits):
        '''
        This function finds the number of cases to be fit.

        Parameters:
        -----------
        NumCases: Integer
                  Number of cases to be considered
        '''

        if (TransitSearch.TransitMatch_Status):
            self.AllCurrentT0s =  TransitSearch.T0s
            self.AllCurrentPeriods = TransitSearch.TP_Periods

            #In order to choose the largest period
            self.AllCurrentSDE =  TransitSearch.SDE + self.AllCurrentPeriods/10000.00

            #Now fit for the transit
            self.CaseNumber=1
            HarmonicNum = 1

            self.FittedT0 = []
            self.FittedPeriod = []

            #Run until number of N best cases are fit or all the cases are adequately considered from SDE.
            while self.CaseNumber<=NumFits or len(self.AllCurrentSDE)<1:
                SelectIndex = np.argmax(self.AllCurrentSDE)
                self.CurrentT0 = self.AllCurrentT0s[SelectIndex]
                self.CurrentPeriod = self.AllCurrentPeriods[SelectIndex]

                self.FittedT0.append(self.CurrentT0)
                self.FittedPeriod.append(self.CurrentPeriod)

                #In order to address problem with negative numbers
                if self.CurrentT0>np.min(Target.AllTime):
                    self.CurrentT0-=self.CurrentPeriod

                #Data length = 1.5 hours around transit
                SelectDataIndex = np.abs((Target.AllTime -self.CurrentT0 +self.TDur/2.)%self.CurrentPeriod)<self.TDur

                #################################################
                self.SelectedTime = Target.AllTime[SelectDataIndex]
                self.SelectedFlux = Target.AllFlux[SelectDataIndex]
                self.SelectedData = Target.ParamValues[SelectDataIndex]


                self.numDataPoints = len(self.SelectedTime)

                #Find the night number()
                self.GetNightNumber(Target)

                print("Now running MCMC for:", self.CaseNumber)
                self.QuickFit(Target, TransitSearch)

                #Fit only if greater than 10
                #Remove certain values around the peak and its harmonic
                self.RemoveIndex = np.zeros(len(self.AllCurrentSDE)).astype(np.bool)
                while self.CurrentPeriod/HarmonicNum>TransitSearch.MinPeriod:
                    self.CurrentRemoveIndex = np.abs(self.AllCurrentPeriods -self.CurrentPeriod/HarmonicNum)<0.070
                    self.RemoveIndex = np.logical_or(self.RemoveIndex, self.CurrentRemoveIndex)
                    HarmonicNum+=1


                self.AllCurrentT0s = self.AllCurrentT0s[~self.RemoveIndex]
                self.AllCurrentPeriods = self.AllCurrentPeriods[~self.RemoveIndex]
                self.AllCurrentSDE = self.AllCurrentSDE[~self.RemoveIndex]



                #The SDE is given by::
                self.CaseNumber+=1


        elif (TransitSearch.TLS_Status):
            print("This will be handled by transit forecaster module.")
        pass


    def Likelihood(self,theta, params):
        '''
        This is likelihood function along with the

        Parameters:
        -----------
        theta: list
                list of fit Parameters. First six are T0, Period, a_Rs, Rp_Rs, b, and u.

        params: batman transit class
                Batman is initialized outside the function to improve the speed.

        '''
        T0 = theta[0]
        Period = theta[1]
        a_Rs = theta[2]
        Rp_Rs = theta[3]
        b = theta[4]
        u = theta[5]

        if np.abs(T0-self.CurrentT0)>self.Tolerance*1.2:
            return -np.inf

        if np.abs(Period-self.CurrentPeriod)>self.Tolerance*1.2:
            return -np.inf

        if b<0.0 or b>1.2:
            return -np.inf

        if u<0.0 or u>1.0:
            return -np.inf

        if a_Rs<3.0:
            return -np.inf

        if Rp_Rs<0.000 or Rp_Rs>0.30:
            return -np.inf

        if max(np.abs(theta[6:]))>1e7:
            return -np.inf


        Inc = np.rad2deg(np.arccos(b/a_Rs))



        params = batman.TransitParams()
        params.t0 = T0                        #time of inferior conjunction
        params.per = Period                   #orbital period
        params.rp = Rp_Rs                     #planet radius (in units of stellar radii)
        params.a = a_Rs                       #semi-major axis (in units of stellar radii)
        params.inc = Inc                      #orbital inclination (in degrees)
        params.ecc = 0.                       #eccentricity
        params.w = 90.                        #longitude of periastron (in degrees)
        params.u = [u]                        #limb darkening coefficients [u]
        params.limb_dark = "linear"


        m = batman.TransitModel(params, self.SelectedTime)    #initializes model
        TransitModelFlux = m.light_curve(params)          #calculates light curve

        DetrendedFlux = np.dot(self.BasisMatrix,theta[6:])
        CombinedModel = DetrendedFlux+TransitModelFlux


        #Number of offsets for number of nights
        StartIndex = 0
        for CurrentNight in range(self.NumNights):
            if CurrentNight == len(self.BreakLocation):
                StopIndex = len(self.SelectedTime)
            else:
                StopIndex = self.BreakLocation[CurrentNight]+1
            Offset =  np.mean(self.SelectedFlux[StartIndex:StopIndex] - CombinedModel[StartIndex:StopIndex])


            #CombinedModel[StartIndex:StopIndex]+=Offset

            #Offset =  np.mean(self.SelectedFlux[StartIndex:StopIndex] - CombinedModel[StartIndex:StopIndex])
            #print("The value of after correction is::", Offset)
            StartIndex = StopIndex



        Residual = self.SelectedFlux - CombinedModel

        SumResidual = np.sum(np.abs(Residual))

        if SumResidual<self.BestResidual:
            self.BestResidual = SumResidual

        ChiSquare = -(0.5*np.sum(Residual*Residual)/(self.STD*self.STD))
        return ChiSquare


    def QuickFit(self, Target, TransitSearch):
        '''
        This method performs the quick fit

        '''

        ########################################################################
        self.SaveLocation = os.path.join(Target.ResultDir, "MCMC_Results")

        if not(os.path.exists(self.SaveLocation)):
            os.system("mkdir %s" %self.SaveLocation)
        ########################################################################


        NCols = 0
        for NightIndex in self.AllNightIndex:
            NCols+=len(TransitSearch.AllCombinationBasis[NightIndex])
        NCols*=2

        #Construct Basis
        self.BasisMatrix = np.zeros((len(self.SelectedTime),NCols))

        StartIndex = 0
        self.NumNights = len(self.BreakLocation) + 1

        AssignCol = 0
        for CurrentNight in range(self.NumNights):
            if CurrentNight == len(self.BreakLocation):
                StopIndex = len(self.SelectedTime)
            else:
                StopIndex = self.BreakLocation[CurrentNight]+1

            for Col in TransitSearch.AllCombinationBasis[CurrentNight]:
                #MeanValue = np.mean(self.SelectedData[StartIndex:StopIndex,Col])
                self.BasisMatrix[StartIndex:StopIndex, AssignCol] = self.SelectedData[StartIndex:StopIndex,Col]
                self.BasisMatrix[StartIndex:StopIndex, AssignCol+1] = np.power(self.SelectedData[StartIndex:StopIndex,Col],2)
                AssignCol += 2
            StartIndex = StopIndex

        nWalkers = int(NCols*8)+20

        self.Parameters = ["T0", "Period", "a_Rs", "Rp_Rs", "b", "q1", "q2"]

        T0_Init = np.random.normal(self.CurrentT0, self.Tolerance/4.0, nWalkers)
        Period_Init = np.random.normal(self.CurrentPeriod, self.Tolerance/4.0, nWalkers)
        a_Rs_Init = np.random.normal(30.0, 3.0, nWalkers)
        Rp_Rs_Init = np.random.normal(0.07, 0.002, nWalkers)
        b_Init = np.random.normal(0.5, 0.01, nWalkers)
        u_Init = np.random.normal(0.5, 0.05, nWalkers)

        Decorrelator = np.random.normal(0,0.01,(nWalkers,NCols))

        StartingGuess = np.column_stack((T0_Init, Period_Init, a_Rs_Init, Rp_Rs_Init, \
                                         b_Init, u_Init, Decorrelator))

        #intiate batman
        params = batman.TransitParams()
        params.limb_dark = "quadratic"       #limb darkening model

        _, nDim = np.shape(StartingGuess)

        self.BestResidual = np.inf

        #input("Before starting MCMC...")
        sampler = emcee.EnsembleSampler(nWalkers, nDim, self.Likelihood, args=[params], threads=8)

        #Run for 500 steps
        state = sampler.run_mcmc(StartingGuess, self.NRuns,  progress=True)
        Probability = sampler.lnprobability


        X,Y = np.where(Probability==np.max(Probability))

        plt.figure(figsize=(14,8))
        plt.plot(-np.mean(Probability, axis=0))
        plt.yscale('log')
        plt.ylabel("Log Probability", fontsize=20)
        plt.xlabel("Step Number", fontsize=20)
        plt.tight_layout()

        if self.SavePlot:
            plt.savefig(os.path.join(self.SaveLocation,"LogProb%s.png"%(str(self.CaseNumber))))
        if self.ShowPlot:
            plt.show()
        plt.close('all')



        #Now construct the best model
        BestTheta = sampler.chain[X[0],Y[0],:]

        BestMCMCT0 = BestTheta[0]
        BestMCMCPeriod = BestTheta[1]
        a_Rs = BestTheta[2]
        Rp_Rs = BestTheta[3]
        b = BestTheta[4]
        u = BestTheta[5]

        Inc = np.rad2deg(np.arccos(b/a_Rs))

        params = batman.TransitParams()
        params.t0 = BestMCMCT0                #time of inferior conjunction
        params.per = BestMCMCPeriod           #orbital period
        params.rp = Rp_Rs                     #planet radius (in units of stellar radii)
        params.a = a_Rs                       #semi-major axis (in units of stellar radii)
        params.inc = Inc                      #orbital inclination (in degrees)
        params.ecc = 0.                       #eccentricity
        params.w = 90.                        #longitude of periastron (in degrees)
        params.u = [u]                   #limb darkening coefficients [u1, u2]
        params.limb_dark = "linear"


        m = batman.TransitModel(params, self.SelectedTime)    #initializes model
        TransitModelFlux = m.light_curve(params)          #calculates light curve

        TrendLine = np.dot(self.BasisMatrix, BestTheta[6:])
        ModelFlux = TransitModelFlux+TrendLine

        #Number of offsets for number of nights
        StartIndex = 0
        Residual = 0
        for CurrentNight in range(self.NumNights):
            if CurrentNight == len(self.BreakLocation):
                StopIndex = len(self.SelectedTime)
            else:
                StopIndex = self.BreakLocation[CurrentNight]+1
            Offset =  np.mean(self.SelectedFlux[StartIndex:StopIndex] - ModelFlux[StartIndex:StopIndex])
            TrendLine+=Offset
            StartIndex = StopIndex

        #Now make the folded plot...
        DetrendedFlux = self.SelectedFlux - TrendLine
        FoldedTime, FoldedFlux = fold_data(self.SelectedTime -BestMCMCT0 + BestMCMCPeriod/2.0, DetrendedFlux, BestMCMCPeriod)
        FoldedTime, FoldedModel = fold_data(self.SelectedTime -BestMCMCT0+ BestMCMCPeriod/2.0, TransitModelFlux, BestMCMCPeriod)
        FoldedErrors = FoldedFlux - FoldedModel

        #Corner plot
        CornerPlotData = sampler.chain[:,self.NRuns//2:,:]
        X,Y,Z = np.shape(CornerPlotData)
        CornerPlotData = CornerPlotData.reshape(X*Y,Z)


        #Construct a corner plot
        corner.corner(CornerPlotData, quantiles=[0.16, 0.5, 0.84])

        SaveName = str(self.CaseNumber).zfill(3)+"_CornerPlot"".png"
        if self.SavePlot:
            plt.savefig(os.path.join(self.SaveLocation , SaveName))
        if self.ShowPlot:
                plt.show()
        plt.close('all')


        #Now the main figure
        NBins = int((FoldedTime[-1]-FoldedTime[0])*60.0*24./(5.0))
        NData = len(FoldedTime)/NBins

        BinnedTime = binned_statistic(FoldedTime, FoldedTime, bins=NBins)[0]
        BinnedFlux = binned_statistic(FoldedTime, FoldedFlux, bins=NBins)[0]
        BinnedError = RunningResidual(FoldedTime, FoldedErrors, NBins)


        print("Saving figures from the MCMC")

        T0Subtract =  BestMCMCPeriod/2.0

        T0Error = np.std(CornerPlotData[:,0])
        PeriodError = np.std(CornerPlotData[:,1])
        Rp_RsError = np.std(CornerPlotData[:,3])

        TitleText = "T0: "+ str(round(BestMCMCT0,5)) + "$\pm$"  + str(round(T0Error,5)) + "\n"+ \
                    "Period: "+ str(round(BestMCMCPeriod,5))+ "$\pm$"  + str(round(PeriodError,5)) + "\n"+ \
                    "Rp_Rs: "+ str(round(Rp_Rs,5))+ "$\pm$"  + str(round(Rp_RsError,5))

        plt.figure(figsize=(14,10))
        plt.plot((FoldedTime-T0Subtract)*24.0, FoldedFlux, color="cyan", marker="o", \
                markersize=2, linestyle="None", zorder=1, label="Detrended Data")
        plt.errorbar((BinnedTime-T0Subtract)*24.0, BinnedFlux, yerr=BinnedError, linestyle="None", \
                     marker="o", markersize=5, capsize=3, elinewidth=2, color="black", zorder=2, label="Binned Data")
        plt.plot((FoldedTime-T0Subtract)*24.0, FoldedModel, "r-", lw=3, label="Model")
        plt.axvline(x=0, color="red", linestyle="-", label="T0")
        YLower, YUpper = np.percentile(FoldedFlux, [2.0, 98.0])
        plt.legend(loc=1)
        plt.ylim([YLower, YUpper])
        plt.ylabel("Normalized Flux", fontsize=20)
        plt.xlabel("Hours since Mid Transit ", fontsize=20)
        plt.title(TitleText)
        SaveName = str(self.CaseNumber).zfill(3)+"Case"".png"
        if self.SavePlot:
            plt.savefig(os.path.join(self.SaveLocation , SaveName))
        if self.ShowPlot:
            plt.show()
        plt.close('all')



        #Save number of fits for individual nights
        StartIndex = 0
        for CurrentNight in range(self.NumNights):
             if CurrentNight == len(self.BreakLocation):
                 StopIndex = len(self.SelectedTime)
             else:
                 StopIndex = self.BreakLocation[CurrentNight]+1

             Time2Plot = self.SelectedTime[StartIndex:StopIndex]
             Flux2Plot = DetrendedFlux[StartIndex:StopIndex]
             Model2Plot = TransitModelFlux[StartIndex:StopIndex]
             Error2Plot = Flux2Plot - Model2Plot

             NBins = int((Time2Plot[-1]-Time2Plot[0])*60.0*24./(5.0))


             if NBins>2:
                 BinnedTime = binned_statistic(Time2Plot, Time2Plot, bins=NBins)[0]
                 BinnedFlux = binned_statistic(Time2Plot, Flux2Plot, bins=NBins)[0]
                 BinnedError = RunningResidual(Time2Plot, Error2Plot, NBins)

             T0Int = int(min(Time2Plot))
             SaveName = "Case"+str(self.CaseNumber).zfill(3)+"_"+str(T0Int) +".png"

             plt.figure(figsize=(14,8))
             plt.plot(Time2Plot-T0Int, Flux2Plot, marker="o", color="cyan", \
                      linestyle="None", label="Data")
             plt.errorbar(BinnedTime-T0Int, BinnedFlux, yerr=BinnedError, linestyle="None", \
                          marker="o", markersize=5, capsize=3, elinewidth=2, color="black", zorder=2, label="Binned Data")
             plt.plot(Time2Plot-T0Int, Model2Plot, "r-", lw=2, label="Model")

             YLower, YUpper = np.percentile(Flux2Plot, [2.0, 98.0])
             plt.ylim([YLower, YUpper])
             plt.ylabel("Normalized Flux", fontsize=20)
             plt.xlabel("JD %s" %(T0Int), fontsize=20)
             plt.legend(loc=1)

             if self.SavePlot:
                 plt.savefig(os.path.join(self.SaveLocation , SaveName))
             if self.ShowPlot:
                 plt.show()
             plt.close('all')

             StartIndex = StopIndex
