#!/usr/bin/python

import os, sys
import subprocess

speaker = 's1'
for i in range(1, 16):
    speaker = 's{0}'.format(i)
    print(speaker)

    # Open files in directory
    path = '../videos'

    for file in os.listdir(os.path.join(path, speaker)):
        if os.path.isfile(os.path.join(path, speaker, file)) and file[-1] == 'g':
            print(file, file[:-4])
            os.mkdir(os.path.join(path, speaker, file[:-4]))
            command = '"C:\Program Files\VideoLAN\VLC\\vlc.exe" -I dummy {0} --video-filter=scene --vout=dummy --scene-ratio=1 --scene-path={1} vlc://quit'.format(os.path.join(path, speaker, file), os.path.join(path, speaker, file[:-4]))
            print(command)
            subprocess.run(command)
