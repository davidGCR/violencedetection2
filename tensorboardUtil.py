import os
import numpy as np
import pandas as pd

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tabulate_events(dpath):

    final_out = {}
    # ea = EventAccumulator(path=dpath)
    for dname in os.listdir(dpath):
        print(f"Converting run {dname}",end="")
        ea = EventAccumulator(os.path.join(dpath, dname)).Reload()
        
        tags = ea.Tags()['scalars']
        print(ea.Tags())

        out = {}

        for tag in tags:
            tag_values=[]
            wall_time=[]
            steps=[]

            for event in ea.Scalars(tag):
                tag_values.append(event.value)
                wall_time.append(event.wall_time)
                steps.append(event.step)

            # out[tag] = pd.DataFrame(data=dict(zip(steps, np.array([tag_values, wall_time]).transpose())), columns=steps, index=['value', 'wall_time'])
            out[tag]=pd.DataFrame(data=dict(zip(steps,np.array([tag_values]).transpose())), columns=steps,index=['value'])

        # print('Len de tags: ',len(tags))
        if len(tags)>0:      
            df= pd.concat(out.values(),keys=out.keys())
            df.to_csv(f'{dname}.csv')
            print("- Done")
        else:
            print('- Not scalers to write')

        final_out[dname] = df
    return final_out

if __name__ == '__main__':
    path = "/Users/davidchoqueluqueroman/Desktop/PAPERS-CODIGOS/violencedetection2/runs/numDynImgs3-segmentLen30-skip2-onlyrawvideos-resnet18-Epochs30"
    steps = tabulate_events(path)
    _, experiment_name = os.path.split(path)
    pd.concat(steps.values(),keys=steps.keys()).to_csv(experiment_name +'.csv')