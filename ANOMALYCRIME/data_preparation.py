import os
import shutil

def get_videos_from_txt(file):
    f = open(file, "r")
    lines = f.readlines()
    f.close()
    lines = [line.rstrip('\n') for line in lines]
    return lines

def search_files(directory_in, directory_out, file, extension):
    extension = extension.lower()
    for dirpath, dirnames, files in os.walk(directory_in):
        for name in files:
            if extension and name.lower().endswith(extension):
                if name == file:
                    found_file = os.path.join(dirpath, name)
                    if not os.path.isfile(os.path.join(directory_out,file)):
                        newPath = shutil.copy(found_file, directory_out)
                        print('saved... ',newPath)
            # elif not extension:
            #     print(os.path.join(dirpath, name))

def __main__():
    complete_dataset = '/media/david/datos/Violence DATA/AnomalyCRIME'
    local_dataset = '/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local/videos'
    lines = get_videos_from_txt('/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local/readme/Videos_from_UCFCrime.txt')
    print(len(lines))
    for line in lines:
        search_files(complete_dataset, local_dataset, line, extension='.mp4')
    

__main__()