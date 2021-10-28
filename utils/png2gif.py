import imageio
import glob
import os
import argparse
import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

parser = argparse.ArgumentParser(description='PNG to GIF')
parser.add_argument('--path', type=str, default='./result/nyc_taxi',
                    help='file path ')
parser.add_argument('--range', nargs='+', type=int,
                    default=[-1],)
args = parser.parse_args()


images = []
filenames = glob.glob(args.path+'/*')

filenames.sort(key=os.path.getmtime)
filenames = [filename for filename in filenames if '.png' in filename]
#filenames.sort(key=alphanum_key)
#print(filenames)
if args.range[0] != -1:
    filenames = filenames[args.range[0]:args.range[1]] # normal 12 dust 10 dark 12 # fire 7
for filename in filenames:
    images.append(imageio.imread(filename))

imageio.mimsave(args.path+'/fig.gif',images,duration=0.05)
