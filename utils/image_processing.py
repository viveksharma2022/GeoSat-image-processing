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
    
    @staticmethod
    def DeterminePaddingArea(shift: tuple, imageShape: np.ndarray)->dict:
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

    def AlignImages(self, dataSupplied = None):
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
        # Loading all bands everytime for each band, to be refactored to reduce memory load latency
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
    
def CrossBandsAlign(bandImages: list)->list:
    """Expects a standardized list of imput images. last image is considered as the reference

    Args:
        bandImages (list): _description_
        bandSeparation (dict): _description_

    Returns:
        list: _description_
    """
    newBandHeight = 0
    totalBands = len(bandImages)
    maxBandHeight = 0
    for data in bandImages:
        maxBandHeight = max(maxBandHeight, data.shape[0])
    newBandHeight += maxBandHeight
    newBandWidth = data.shape[1] # width of bands are same

    interBandAlignedData = []
    offsets = []
    tForms = []

    maxPadLeft, maxPadRight, maxPadTop, maxPadBottom = 0, 0 ,0 ,0
    fixedNorm = cv2.normalize(bandImages[totalBands - 1], None, 0 , 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    for i in range(len(bandImages)):
        movingNorm = cv2.normalize(bandImages[i], None, 0 , 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        minHeight = min(fixedNorm.shape[0], movingNorm.shape[0])
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(fixedNorm)
        # plt.subplot(1,2,2)
        # plt.imshow(movingNorm)
        # plt.show()

        offset, tform = ComputeOrbShift(fixedNorm[:minHeight,:], movingNorm[:minHeight,:])
        offsets.append(offset)
        tForms.append(tform)

        paddingProps = ImageAlignmentStrategy.DeterminePaddingArea(offset,bandImages[i].shape)
        maxPadLeft = max(maxPadLeft, paddingProps["padLeft"])
        maxPadRight = max(maxPadRight, paddingProps["padRight"])
        maxPadTop = max(maxPadTop, paddingProps["padTop"])
        maxPadBottom = max(maxPadBottom, paddingProps["padBottom"])

    # warp images to align all the bands together
    interBandAlignedData = []
    cW, cH = newBandWidth + maxPadLeft + maxPadRight + 1, newBandHeight + maxPadTop + maxPadBottom + 1
    for i in range(len(bandImages)):
        canvas = np.zeros((cH, cW))
        h, w = bandImages[i].shape
        canvas[maxPadTop:maxPadTop+h, maxPadLeft:maxPadLeft+w] = bandImages[i]
        canvas = cv2.warpAffine(canvas, tForms[i], (cW, cH))
        interBandAlignedData.append(canvas)

    return interBandAlignedData


def ComputeOrbShift(img1, img2):
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # ORB detector
    orb = cv2.ORB_create(5000)

    # Detect and compute descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        raise ValueError("Could not find descriptors in one of the images.")

    # Match with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract point coordinates from matches
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Estimate translation using RANSAC or least-squares
    if len(pts1) >= 4:
        transform_matrix, _ = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC)
        if transform_matrix is not None:
            dx, dy = transform_matrix[0, 2], transform_matrix[1, 2]
            return (-dx, -dy), transform_matrix
    else:
        raise ValueError("Not enough matches to estimate shift.")

    return (0, 0), None

def StandardizeData(data:list)->list:
    standardizedData = []
    for dat in data:
        meanVal = np.mean(dat)
        stdVal = np.std(dat)
        standardizedData.append((dat - meanVal)/ (stdVal + 1e-8))
    return standardizedData

def NormalizeSNR(bandData:list, panBandData: np.ndarray, normFactor = 0.9):

    def GetSNR(data:np.ndarray)->np.float32:
        meanVal = np.nanmean(data)
        stdVal = np.nanstd(data)
        return stdVal/meanVal, meanVal, stdVal
    
    # compute pandBandSNR
    panSNR, _, _ = GetSNR(panBandData)
    targetSNR = normFactor*panSNR

    # Scale SNR for all bands
    snrNormData = []
    for dat in bandData:
        snr, meanV, stdV = GetSNR(dat)
        scale = targetSNR/snr
        datSNRNorm = (dat - meanV) * scale + meanV
        snrNormData.append(datSNRNorm)

    return snrNormData

if __name__ == "__main__":

    # granular testing, not production ready, just adding path for quick testing, to be refactored later
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    __package__ = 'utils'
    import utils

    params = {"dataFolder": r"C:\Users\vivek\Documents\Project\geoSat\Kaleideo IPS Dataset-20250523T194008Z-1-001\Kaleideo IPS Dataset",
              }
    
    imgAlignStrategy = ImageAlignmentStrategy(PhaseCorrelation(), params["dataFolder"]) # set the strategy to align images
    # rawData = imgAlignStrategy.GetRawData(bandNumber = bandNumber)

    intraBandAlignedData = imgAlignStrategy.AlignImages()

    # # Visualize aligned bands
    # plt.figure(figsize = (10,30))
    # for bandId in range(utils.config.GEOTIFFIMAGE_BANDS):
    #     plt.subplot(utils.config.GEOTIFFIMAGE_BANDS, utils.config.GEOTIFFIMAGE_BANDS//2, bandId+1)
    #     plt.imshow(intraBandAlignedData[bandId], cmap = 'gray')
    #     plt.clim([0,1.2E4])
    #     plt.title(f"BandID: {bandId}")
    #     plt.colorbar()
    #     plt.tight_layout()
    # plt.show()

    intraBandAlignedDataSNRNorm = NormalizeSNR(intraBandAlignedData, intraBandAlignedData[-1])

    # plt.figure()
    # for idx, dat in enumerate(intraBandAlignedDataSNRNorm):
    #     plt.subplot(5,2,idx+1)
    #     plt.imshow(dat)
    #     plt.clim([0,1E4])
    #     plt.colorbar()
    #     plt.tight_layout()
    # plt.show()

    interBandAlignedData = CrossBandsAlign(intraBandAlignedDataSNRNorm)
    

    # Extract the RGB band and create a composite
    rgbMosaic = []
    for band in ["RED", "GREEN", "BLUE"]:
        bandData = cv2.normalize(interBandAlignedData[utils.config.BANDS[band]], None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        rgbMosaic.append(bandData)

    mosaic = np.stack(rgbMosaic, axis = -1)
    plt.figure()
    plt.imshow( mosaic, cmap='jet')
    plt.show()

    plt.figure()
    plt.imshow(intraBandAlignedData[-1])
    plt.clim([0, 1E4])
    plt.show()

    plt.figure()
    plt.imshow(intraBandAlignedData[0])
    plt.clim([0, 1E4])
    plt.show()

