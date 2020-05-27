import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse
import matplotlib.pyplot as plt
# import constants

def tabulate_events(dpath, dout, save):
    final_out = {}
    # for dname in os.listdir(dpath):
    dname = os.listdir(dpath)[0]
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
        
        out[tag] = pd.DataFrame(data=dict(zip(steps, np.array([tag_values]).transpose())), columns=steps, index=None)
        # print(out[tag].head(10))

    if len(tags)>0:      
        df = pd.concat(out.values())

        # print('values: ',out.keys())
        if save:
            df.to_csv(f'{os.path.join(dout,dname)}.csv')
        else:
            df_T = df.transpose()
            df_T.columns = out.keys()
            # print(df_T.head(10))
            x = df_T.index.values.tolist()
            train_loss = df_T['training loss'].to_numpy()
            val_loss = df_T['validation loss'].to_numpy()
            train_acc = df_T['training Acc'].to_numpy()
            val_acc = df_T['validation Acc'].to_numpy()
            
            # plt.figure()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,4), sharex=True)
            plt.xticks(np.arange(min(x), max(x), 2))

            ax1.plot(x, train_loss, 'r', label='train')
            ax1.plot(x, val_loss, 'b', label='val')
            ax1.set_title('Loss curves')
            legend = ax1.legend(loc='upper right', shadow=True, fontsize='large')
            

            ax2.plot(x, train_acc, 'r', label='train')
            ax2.plot(x, val_acc,'b', label='val')
            ax2.set_title('Accuracy curves')
            legend = ax2.legend(loc='lower right', shadow=True, fontsize='large')
            
            # plt.figure(figsize=(3, 8))
            
            if not os.path.exists('tmp_images'):
                os.mkdir('tmp_images')
            fig.savefig('tmp_images/tmp.png')
            plt.show()
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--folderIn", type=str)
    parser.add_argument("--folderOut", type=str, default='')
    parser.add_argument("--save", type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()
    # folderIn = 'RESULTS/Vif-tensorboard/VIF-Model-resnet50, segmentLen-26, numDynIms-1, frameSkip-0, epochs-25, splitType-cross-val, fold-1'
    # folderIn = 'RESULTS/HOCKEY/tensorboard-runs/HOCKEY-Model-alexnet, segmentLen-20, numDynIms-1, frameSkip-0, epochs-25, split_type-train-test'
    # folderOut = 'RESULTS/HOCKEY/learningCurves'
    # steps = tabulate_events(folderIn)
    path, file_name = os.path.split(args.folderIn)
    final = tabulate_events(path, dout=args.folderOut, save=args.save)
    # final.to_csv(os.path.join(folderOut, file_name[:-1]+'.csv'))
    # pd.concat(steps.values()).to_csv(os.path.join(folderOut, file_name+'.csv'))
    # pd.concat(steps.values(),keys=steps.keys())