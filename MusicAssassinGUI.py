import time
from tkinter import *
from PIL import ImageTk, Image
import os
import numpy as np
#from tqdm import tqdm


from spleeter.separator import Separator
from moviepy.editor import VideoFileClip
from moviepy.audio.AudioClip import AudioArrayClip
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

# def requirements():
#     os.system('cmd /c "pip install mediainfo"')
#     os.system('cmd /c "pip install ffmpeg"')
#     os.system('cmd /c "pip install spleeter"')
    
# requirements

print("Setting up GUI...")
#Setting up GUI
#Window or Display Size
root = Tk()
root.wm_title("Music Assassin")
root.geometry("400x450")
root.configure(background = 'white')


#Label Functions
def Labels():
    title = Label(root, text = "Music Assassin", font = "Verdana 18 bold", bg='white')  
    title.pack(side=TOP, expand=None, fill=X, padx=(10, ), pady=(20, 0)) 
    
    credits = Label(root, text = "—A project of mAb Engineers",font = "Verdana 6", bg='white')  
    credits.pack(side=TOP, expand=None, fill=X, padx=(10, 20), pady=(0, 0))
    
    prompt = Label(root, text = "Choose the video to de-music:",font = "Verdana 13 bold", bg='white')  
    prompt.pack(side=TOP, expand=None, fill=X, padx=(10, 20), pady=(7, 10))
    
    left = Label(root, text = "   ", font = "Verdana 18 bold", bg='white')  
    left.pack(side=LEFT, expand=None, fill=Y)
    right = Label(root, text = "   ", font = "Verdana 18 bold", bg='white')  
    right.pack(side=RIGHT, expand=None, fill=Y)
    

def get_file_path():
    global browse
    # Open and return file path
    browse = filedialog.askopenfilename(title = "Select A File", filetypes = (("mp4", "*.mp4"), ("mov files", "*.png"), ("wmv", "*.wmv"), ("avi", "*.avi")), multiple=True)

    
def demusic(file_path):
    """ testmoviepy """

    
    # Making names of files
    file_path = file_path.replace("\\", "/")
    name = (((file_path.split("/"))[-1]).split("."))[0]
    path = "/".join(file_path.split("/")[:-1])
    video_name = name + "-no-music.mp4"

    # Loading audio and video
    print("Loading audio and video:")
    clip = VideoFileClip(file_path)
    waveform = clip.audio.to_soundarray()
    sample_rate = clip.audio.fps

    # Spliting audio into smaller parts
    print("Spliting audio into smaller parts:")
    t = int(sample_rate)*60 # s
    no_split = np.ceil(waveform.shape[0]/t)
    wave = np.array_split(waveform,no_split, axis=0)

    # Perform the separation :
    print("Starting Seperation")
    print("No. Sections:",len(wave))
    separator = Separator('spleeter:2stems')
    no_music = np.zeros(waveform.shape)
    start = 0
    for i,wav in enumerate(wave):
        print("Section:",i)
        pred = separator.separate(wav)
        no_music[start:start+wav.shape[0],:]= pred['vocals']
        start += wav.shape[0]

    # Joining audio and video
    print("Rejoining audio and video:")
    audio_no_music = AudioArrayClip(no_music, fps=sample_rate)
    no_music_clip = clip.set_audio(audio_no_music)

    # Exporting
    print("Exporting product:")
    no_music_clip.write_videofile(path+'/'+video_name)

    print('Done')
    
def demusic_list():
    global files
    j = 1
    for i in files:
        demusic(i)
        j += 1
    
    d2 = Label(root, text = "Done", wraplength=300, bg="black", fg="white", justify="left").pack(side=TOP, expand=None, fill=X)
    files = []

#Title Labels
Labels()

files = []
file_path = ""
browse = [""]
b1 = Button(text = "Browse", command = get_file_path, width = 15, bg="black", fg="white",font = "Verdana 10 bold").pack(side=TOP, expand=None, fill=X)
f1 = Button(text = "Demusic", command = demusic_list, width = 15, bg="black", fg="white",font = "Verdana 10 bold").pack(side=TOP, expand=None, fill=X)

while 1:
    for i in browse:
        file_path = i
        if file_path not in files and file_path != "":
            l1 = Label(root, text = "File path: \n" + str(file_path), wraplength=300, justify="left").pack(side=TOP, expand=None, fill=X)
            print(file_path)
            root.update()
            files.append(file_path)
    browse = [""]
    root.update()

print("...\n")

print('GUI Setup completed')
root.mainloop()
