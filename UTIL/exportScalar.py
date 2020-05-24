import os
import numpy as np
import pandas as pd

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse

def tabulate_events(dpath):
    final_out = {}
    for dname in os.listdir(dpath):
        print(f"Converting run {dname}",end="")
        ea = EventAccumulator(os.path.join(dpath, dname)).Reload()
        tags = ea.Tags()['scalars']

        out = {}

        for tag in tags: #training_loss validation_loss training_Acc validation_Acc
            tag_values=[]
            # wall_time=[]
            steps=[]
            for event in ea.Scalars(tag):
                tag_values.append(event.value)
                # wall_time.append(event.wall_time)
                steps.append(event.step)

            # out[tag]=pd.DataFrame(data=dict(zip(steps,np.array([tag_values,wall_time]).transpose())), columns=steps,index=['value','wall_time'])
            out[tag] = pd.DataFrame(data=dict(zip(steps, np.array([tag_values]).transpose())), columns=steps,index=['value'])
            # print(out[tag].head(10))

        if len(tags)>0:      
            df = pd.concat(out.values(), keys=out.keys())
            # print(df.head(10))
            # df.to_csv(f'{dname}.csv')
            print("- Done")
        else:
            print('- Not scalers to write')

        final_out[dname] = df


    return final_out

def tabulate_events_kfolds(dpath):
    fff = []
    final_out = {}
    ll = os.listdir(dpath)
    ll.sort()
    for i, fold in enumerate(ll):
        
        for dname in os.listdir(os.path.join(dpath,fold)):
            print(f"Converting run {os.path.join(dpath,fold, dname)}",end="")
            ea = EventAccumulator(os.path.join(dpath, fold, dname)).Reload()
            tags = ea.Tags()['scalars']

            out = {}

            for tag in tags:
                tag_values=[]
                # wall_time=[]
                steps=[]

                for event in ea.Scalars(tag):
                    tag_values.append(event.value)
                    # wall_time.append(event.wall_time)
                    steps.append(event.step)

                # out[tag]=pd.DataFrame(data=dict(zip(steps,np.array([tag_values,wall_time]).transpose())), columns=steps,index=['value','wall_time'])
                out[tag]=pd.DataFrame(data=dict(zip(steps,np.array([tag_values]).transpose())), columns=steps)

            if len(tags)>0:      
                df= pd.concat(out.values(),keys=out.keys())
                # df.to_csv(f'{dname}.csv')
                print("- Done")
            else:
                print('- Not scalers to write')

            final_out[str(i+1)] = df
        
    fff = pd.concat(final_out.values(), keys=final_out.keys())
    return fff

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--folderIn", type=str)
    # parser.add_argument("--folderOut", type=str)
    # args = parser.parse_args()
    folderIn = 'RESULTS/Vif-tensorboard/VIF-Model-resnet50, segmentLen-26, numDynIms-1, frameSkip-0, epochs-25, splitType-cross-val, fold-1'
    folderOut = 'RESULTS/VIF/learningCurves'
    # steps = tabulate_events(folderIn)
    path, file_name = os.path.split(folderIn)
    final = tabulate_events_kfolds(path)
    final.to_csv(os.path.join(folderOut, file_name[:-1]+'5Folds.csv'))
    # pd.concat(steps.values()).to_csv(os.path.join(folderOut, file_name+'.csv'))
    # pd.concat(steps.values(),keys=steps.keys())