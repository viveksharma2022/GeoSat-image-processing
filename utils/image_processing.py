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
        # calculte shift
        return cv2.phaseCorrelate(fixed, moving)


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
        from utils import io
        alignedImages = []
        fixed = None  # normalized fixed for feature detection

        for idx, data in enumerate(io.LoadBulkData(self.dataFolder, '*.tiff')):
              # original image, possibly float32 or original dtype
            # norm_img = cv2.normalize(original_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # normalized uint8 for features
            curImg = data.data[-1].astype(np.float32)
            if idx == 0:
                fixed = curImg
                alignedImages.append(fixed)  # store original first frame as is
            else:
                moving = curImg
                shift, response = self.registrationManager.Register(fixed, moving)
                
                paddingProps, fixedPadded, alignedPadded = self.AlignFixedAndMoving(fixed, moving, shift)
                self.VisualizeOverlapArea(fixedPadded,alignedPadded)

                pause = 1



        return np.vstack(alignedImages)
        

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

                # orb = cv2.ORB_create(nfeatures=1000)
                # kp1, des1 = orb.detectAndCompute(fixed_norm, None)
                # kp2, des2 = orb.detectAndCompute(moving_norm, None)

                # if des1 is None or des2 is None:
                #     raise RuntimeError("No descriptors found.")

                # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                # matches = matcher.match(des1, des2)
                # matches = sorted(matches, key=lambda x: x.distance)

                # if len(matches) < 4:
                #     raise RuntimeError("Not enough matches for reliable registration.")

                # pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
                # pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

                # affine_transform, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC)

                # if affine_transform is None:
                #     raise RuntimeError("Could not estimate transform.")

                # # Enforce rigid transform by removing scaling/shearing
                # R = affine_transform[:, :2]
                # t = affine_transform[:, 2]

                # U, _, Vt = np.linalg.svd(R)
                # R_rigid = np.dot(U, Vt)
                # rigid_transform = np.hstack([R_rigid, t.reshape(2, 1)])

                # height, width = fixed_norm.shape

                # # Warp the original image (not normalized) using the rigid transform
                # aligned_original = cv2.warpAffine(original_img, rigid_transform, (width, height),
                #                                 flags=cv2.INTER_LINEAR,
                #                                 borderMode=cv2.BORDER_REFLECT)

                # alignedImages.append(aligned_original)

                # # Update fixed_norm for next iteration: normalize the aligned original image
                # fixed_norm = cv2.normalize(aligned_original, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)




    #     def calculate_shifts(frames):
    #     shifts = []
    #     prev = None
    #     for frame in frames:
    #         gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         gray = gray.astype(np.float32)
    #         if prev is not None:
    #             shift, _ = cv2.phaseCorrelate(prev, gray)
    #             shifts.append(shift)  # (dx, dy)
    #         prev = gray
    #     return shifts

    # def align_frames(frames, shifts):
    #     aligned_frames = [frames[0]]
    #     total_dx, total_dy = 0, 0

    #     for i, (dx, dy) in enumerate(shifts):
    #         total_dx += dx
    #         total_dy += dy

    #         M = np.float32([[1, 0, -total_dx],
    #                         [0, 1, -total_dy]])

    #         aligned = cv2.warpAffine(frames[i + 1], M, (frames[0].shape[1], frames[0].shape[0]))
    #         aligned_frames.append(aligned)

    #     return aligned_frames