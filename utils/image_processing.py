from abc import ABC, abstractmethod
import numpy as np

class Registration:
    def __init__(self):
        self.aboutMe = "Registration interface"
        self.rawData = np.nan
        self.alignedData = np.nan


            
    @abstractmethod
    def Register(self, fixed:np.ndarray, moving:np.ndarray):
        pass

class PhaseCorrelation(Registration):
    def __init__(self):
        super().__init__()
        self.aboutMe = "Registration using phase correlation"

    def Register(self, fixed:np.ndarray, moving:np.ndarray):
        # calculte shift
        shift, _ = cv2.phaseCorrelate(fixed, moving)
        pass

class ImageAlignmentStrategy:
    def __init__ (self, registrationStrategy:Registration, dataFolder:str):
        self.registrationStrategy = registrationStrategy
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

    def AlignImages(self):
        from utils import io
        alignedImages = []
        fixed, moving = np.nan, np.nan
        for idx, data in enumerate(io.LoadBulkData(self.dataFolder, '*.tiff')):
            if idx == 0:
                fixed = data.data[-1]
            else:
                moving = data.data[-1]

            self.registrationStrategy.Register(fixed, moving)

            pass
        

    

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