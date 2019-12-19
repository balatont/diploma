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
            command1 = '"C:\Program Files\VideoLAN\VLC\\vlc.exe" {0} '.format(os.path.join(path, speaker, file))
            command2 = ' -I dummy --no-sout-video --sout-audio --no-sout-rtp-sap --no-sout-standard-sap --ttl=1 --sout-keep '
            command3 = '--sout "#transcode{acodec=s16l,channels=2}:std{access=file,mux=wav,dst='
            command4 = '{0}.wav'.format(os.path.join(path, speaker, file[:-4], file[:-4]))
            command5 = '}" vlc://quit'

            command = command1 + command2 + command3 + command4 + command5
            print(command)
            subprocess.call(command)
