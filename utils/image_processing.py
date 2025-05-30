from abc import ABC, abstractmethod
import numpy as np
import cv2
import sys, os
import matplotlib.pyplot as plt

class Registration:
    def __init__(self):
        self.aboutMe = "Registration interface"
            
    @abstractmethod
    def Register(self, fixed:np.ndarray, moving:np.ndarray):
        pass

class PhaseCorrelation(Registration):
    def __init__(self):
        super().__init__()
        self.aboutMe = "Registration using phase correlation"

    def Register(self, fixed:np.ndarray, moving:np.ndarray):

        if(fixed.shape != moving.shape):
            raise Exception (f"[ERROR]: fixed and moving shape must be same")

        # Generate Hamming window
        winY = np.hamming(fixed.shape[0])
        winX = np.hamming(fixed.shape[1])
        window = np.outer(winY, winX)

        # calculte shift
        return cv2.phaseCorrelate(fixed * window, moving * window)


class ImageAlignmentStrategy:
    def __init__ (self, registrationStrategy:Registration, dataFolder:str):
        self.registrationManager = registrationStrategy
        self.dataFolder = dataFolder
    
    def GetRawData(self, bandNumber: int):
        from utils import io
        rawData = []
        for data in io.LoadBulkData(self.dataFolder, '*.tiff'):
            try:
                rawData.append(data.data[bandNumber])
            except Exception as e:
                print(f"[ERROR]: Loading data {data.dataPath}, {e}")
        return np.vstack(rawData)
    
    def DeterminePaddingArea(self, shift: tuple, imageShape: np.ndarray)->dict:
        """_summary_

        Args:
            shift (tuple): _description_
            imageShape (np.ndarray): _description_

        Returns:
            dict: _description_
        """        
        dx, dy = shift
        h,w = imageShape 
        padding = dict()
        padding["padLeft"], padding["padRight"] = int(max(0,dx)), int(max(0,-dx))
        padding["padTop"], padding["padBottom"] = int(max(0,dy)), int(max(0,-dy))
        padding["newWidth"] = w + padding["padLeft"] + padding["padRight"]
        padding["newHeight"] = h + padding["padTop"] + padding["padBottom"]
        return padding
    
    def CreateCanvas(self, data:np.ndarray, paddingProps:dict, originalImageSize: tuple) -> np.ndarray:
        """_summary_

        Args:
            data (np.ndarray): _description_
            paddingProps (dict): _description_
            originalImageSize (tuple): _description_

        Returns:
            np.ndarray: _description_
        """        
        h,w = originalImageSize
        dataPadded = np.zeros((paddingProps["newHeight"],paddingProps["newWidth"]), dtype = np.float32)
        dataPadded[paddingProps["padTop"]:paddingProps["padTop"]+h, paddingProps["padLeft"]:paddingProps["padLeft"]+w] = data
        return dataPadded
    
    def VisualizeOverlapArea(self, fixed, aligned, alpha = 0.5):
        fixedNorm = cv2.normalize(fixed, None, 0.0, 1.0, cv2.NORM_MINMAX)
        alignedNorm = cv2.normalize(aligned, None, 0.0, 1.0, cv2.NORM_MINMAX)
        
        mergedRGB = np.stack([fixedNorm,alpha*alignedNorm,np.zeros_like(fixed)], axis = -1)
        fixedRGB = np.stack([fixedNorm,np.zeros_like(fixed),np.zeros_like(fixed)], axis = -1)
        alignedRGB = np.stack([np.zeros_like(fixed),alpha*alignedNorm, np.zeros_like(fixed)], axis = -1)

        plt.figure()
        plt.subplot(3,1,1)
        plt.imshow(fixedRGB)
        plt.title("Fixed")

        plt.subplot(3,1,2)
        plt.imshow(alignedRGB)
        plt.title("Moving")
        
        plt.subplot(3,1,3)
        plt.imshow(mergedRGB)
        plt.title("Registered")
        plt.show()
        
    def AlignFixedAndMoving(self, fixed, moving, shift):

        paddingProps = self.DeterminePaddingArea(shift,moving.shape)
        alignedPadded = self.CreateCanvas(moving, paddingProps, moving.shape)
        # Build translation matrix to shift within new canvas
        dx, dy = shift
        M = np.float32([
            [1, 0, -dx],
            [0, 1, -dy]
        ])
        alignedPadded = cv2.warpAffine(alignedPadded, M, (paddingProps["newWidth"], paddingProps["newHeight"]))
        fixedPadded = self.CreateCanvas(fixed, paddingProps, moving.shape)
        return paddingProps, fixedPadded, alignedPadded

    def AlignImages(self):
        from utils import io, config

        print(f"[INFO]: Aligning images")

        # alignment Lists
        paddingLeft, paddingRight = [0], [0]
        paddingTop, paddingBottom = [0], [0]
        shiftX, shiftYAcc = [0], [0]
        maxPadLeft, maxPadRight, maxPadBottom = 0, 0, 0

        panBandID = config.GEOTIFFIMAGE_BANDS-1 # PANBand is used to determine shifts of frames as it is the one with maximum amount of incident radiation
        for idx, data in enumerate(io.LoadBulkData(self.dataFolder, '*.tiff')): 
            try:
                curImg = np.flipud(data.data[panBandID]).astype(np.float32)
                if idx == 0:
                    fixed = curImg
                else:
                    moving = curImg
                    shift, response = self.registrationManager.Register(fixed, moving)                
                    paddingProps = self.DeterminePaddingArea(shift,moving.shape)
                    
                    # padding and shift values cumulated
                    paddingLeft.append(paddingProps["padLeft"])
                    paddingRight.append(paddingProps["padRight"])
                    paddingTop.append(paddingProps["padTop"])
                    paddingBottom.append(paddingProps["padBottom"])
                    shiftX.append(int(shift[0]))
                    shiftYAcc.append(shiftYAcc[-1] - int(shift[1])) # shift accumulation along Y
                    
                    # max padding values to determine size of total canvas
                    maxPadLeft = max(maxPadLeft, paddingProps["padLeft"])
                    maxPadRight = max(maxPadRight, paddingProps["padRight"])
                    maxPadBottom = max(maxPadBottom, paddingProps["padBottom"])

                    fixed = moving
            except Exception as e:
                print(f"[ERROR]: imageID: {idx}, bandID: {panBandID}, {e}")
        
        # Align the images based on the calculated shifts
        numImages = idx+1
        canvasBands = []
        # Determine the shifts of consecutive images
        # Loading all bands everytime for each band, idea is to reduce the memory footprint of alignment lists, a design decision to be made based on priority:
        # reduce memory footprint or memory load latency
        for band in range(config.GEOTIFFIMAGE_BANDS):
            bandHeight, bandWidth = data.data[band].shape
            canvas = np.zeros((bandHeight + shiftYAcc[-1], bandWidth + maxPadLeft + maxPadRight), dtype = np.float32) # create a canvas to hold all the frames
            for idx, data in enumerate(io.LoadBulkData(self.dataFolder, '*.tiff')):
                try:
                    curImg = np.flipud(data.data[band]).astype(np.float32)
                    h = shiftYAcc[idx]
                    w = maxPadLeft - shiftX[idx]
                    canvas[h:h+bandHeight, w:w+bandWidth] = np.maximum(canvas[h:h+bandHeight, w:w+bandWidth], curImg) # take maximum of the overlap area
                except Exception as e:
                    print(f"[ERROR]: imageID: {idx}, bandID: {band}, {e}")

            canvasBands.append(canvas)
        return canvasBands

if __name__ == "__main__":

    # granular testing, not production ready, just adding path for quick testing, to be refactored later
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    __package__ = 'utils'

    params = {"dataFolder": r"C:\Users\vivek\Documents\Project\geoSat\Kaleideo IPS Dataset-20250523T194008Z-1-001\Kaleideo IPS Dataset",
              }
    
    imgAlignStrategy = ImageAlignmentStrategy(PhaseCorrelation(), params["dataFolder"]) # set the strategy to align images
    # rawData = imgAlignStrategy.GetRawData(bandNumber = bandNumber)

    alignedData = imgAlignStrategy.AlignImages()

    plt.figure()
    plt.imshow(alignedData[-1])
    plt.clim([0, 1E4])
    plt.show()

    plt.figure()
    plt.imshow(alignedData[0])
    plt.clim([0, 1E4])
    plt.show()
