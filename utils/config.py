GEOTIFFIMAGE_BANDS = 5
GEOTIFFIMAGE_SEPERATION = dict({0: [0, 113],
                               1: [114,221],
                               2: [222,329],
                               3: [330,437],
                               4: [438,726]
                               })

MOSAIC_IMAGE_SHAPE_XY = (2355, 6500) # Cols, Rows: size of each band image
BANDS = {"NIR": 0,
        "BLUE": 1,
        "RED EDGE": 2,
        "RED": 3,
        "GREEN": 4}

SENSOR_DISTANCE = {0: 1427,
                   1: 1135,
                   2: 825,
                   3: 492,
                   4: 0}