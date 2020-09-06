import argparse

class Arguments():
    def __init__(self, **kwargs):
        self.parser = argparse.ArgumentParser()
        if ids is None:
            self.defaultArguments()
        else:
            assert len(ids) == len(values), "Error: Parser Arguments and Values have different length."
            for i, item in enumerate(ids):
                parser.add_argument(item, type=type(values[i]).__name__)
        self.args = self.parser.parse_args()

    
    def defaultArguments(self):
        self.parser.add_argument("--modelType",type=str,default="alexnet",help="model")
        self.parser.add_argument("--numEpochs",type=int,default=30)
        self.parser.add_argument("--batchSize",type=int,default=64)
        self.parser.add_argument("--featureExtract",type=lambda x: (str(x).lower() == 'true'), default=False, help="to fine tunning")
        self.parser.add_argument("--numDynamicImagesPerVideo", type=int)
        self.parser.add_argument("--joinType", type=str)
        self.parser.add_argument("--foldsNumber", type=int, default=5)
        self.parser.add_argument("--numWorkers", type=int, default=4)
        self.parser.add_argument("--videoSegmentLength", type=int)
        self.parser.add_argument("--positionSegment", type=str)
        self.parser.add_argument("--split_type", type=str)
        self.parser.add_argument("--overlapping", type=float)
        self.parser.add_argument("--frameSkip", type=int, default=0)
        self.parser.add_argument("--patience", type=int, default=5)
        self.parser.add_argument("--skipInitialFrames", type=int, default=0)
        self.parser.add_argument("--transferModel", type=str, default=None)
        self.parser.add_argument("--saveCheckpoint", type=lambda x: (str(x).lower() == 'true'), default=False)
        self.parser.add_argument("--segmentPreprocessing", type=lambda x: (str(x).lower() == 'true'), default=False)
