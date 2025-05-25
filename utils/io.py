import numpy as np
import cv2
import matplotlib.pyplot as plt
import config
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Iterator

class GeoFileData:
    def __init__(self, bands = 5):
        self.dataPath = ""
        self.rawWidth = []
        self.rawHeight = []
        self.bands = config.GEOTIFFIMAGE_BANDS
        self.data = np.nan
        self.rawData = np.nan

    def _SegmentBands(self, data:np.ndarray) -> np.ndarray:
        """Segments the data into bands, that are fused along the rows

        Args:
            data (np.ndarray): fused numpy array
        Returns:
            np.ndarray: fused array segmented along dimension 0
        """        
        height, width = data.shape[0], data.shape[1]
        bandHeight = height//self.bands
        dataBands = []
        for i in range(self.bands):
            start = config.GEOTIFFIMAGE_SEPERATION[i][0]
            end = min(height, config.GEOTIFFIMAGE_SEPERATION[i][1]) 
            dataBands.append(data[start:end,:])
        return dataBands

    def ReadFile(self, dataPath:str):
        """Reads contents of the geosat data stored as a tiff file
        Args:
            dataPath (str): path to the data
        """        
        self.dataPath = dataPath
        self.rawData = cv2.imread(dataPath, cv2.IMREAD_UNCHANGED)
        self.rawHeight, self.rawWidth = self.rawData.shape[0], self.rawData.shape[1]
        self.data = self._SegmentBands(self.rawData)

    def DataPlainView(self):
        """Plots images of raw data and the segmented bands
        """        
        # === First Figure: Raw Data with matching colorbar height ===
        fig1 = plt.figure(figsize=(6, 5))
        gs1 = gridspec.GridSpec(1, 2, width_ratios=[20, 1])

        ax_img = fig1.add_subplot(gs1[0])
        im = ax_img.imshow(self.rawData, cmap='gray')
        ax_img.set_title('Raw data')

        # Colorbar with matched height
        ax_cb = fig1.add_subplot(gs1[1])
        cbar = plt.colorbar(im, cax=ax_cb)
        ax_cb.set_aspect('auto')  # makes sure it matches the image height
        plt.show()

        # === Second Figure: Band images with individual colorbars ===
        num_bands = config.GEOTIFFIMAGE_BANDS
        fig2 = plt.figure(figsize=(6, 3 * num_bands))
        gs2 = gridspec.GridSpec(num_bands, 2, width_ratios=[20, 1], hspace=0.4)

        for idx in range(num_bands):
            ax_img = fig2.add_subplot(gs2[idx, 0])
            im = ax_img.imshow(self.data[idx])
            ax_img.set_title(f'Band: {idx}')

            ax_cb = fig2.add_subplot(gs2[idx, 1])
            plt.colorbar(im, cax=ax_cb)
            ax_cb.set_aspect('auto')  # Ensure same height as image
        plt.show()

def LoadBulkData(folder: str, patternMatch:str)->Iterator[GeoFileData]:
    """Generates an iterator to read all the geo files in the folder

    Args:
        folder (str): path to the folder containing the files
        patternMatch (str): regular ecpression match for the file

    Yields:
        Iterator[GeoFileData]: iterator containing geodata objects
    """    
    for file in Path(folder).glob(patternMatch):
        data = GeoFileData()
        data.ReadFile(str(file))
        yield data

if __name__ == "__main__":

    dataPath = r"C:\Users\vivek\Documents\Project\geoSat\Kaleideo IPS Dataset-20250523T194008Z-1-001\Kaleideo IPS Dataset\image_003772.tiff" # modify the path
    data = GeoFileData()
    data.ReadFile(dataPath)
    data.DataPlainView()



