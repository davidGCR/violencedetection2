class ArgParameters():
    def __init__(self, ):
        args = {
        'X': datasetAll,
        'y': labelsAll,
        'numFrames': numFramesAll,
        'transform': None,
        'NDI': 1,
        'videoSegmentLength': 20,
        'positionSegment': 'begin',
        'overlapping': 0,
        'frameSkip': 0,
        'skipInitialFrames': 0,
        'batchSize': 8,
        'shuffle': False,
        'numWorkers': 4,
        'pptype': None, 
    }