""" GUI - With logger """

# For GUI
import time
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import os
#from tqdm import tqdm

# For Logger
import sys
from tkinter import Tk, Button, Frame
from tkinter.scrolledtext import ScrolledText

# For Demusic
from spleeter.separator import Separator
from moviepy.editor import VideoFileClip
from moviepy.editor import AudioFileClip
from moviepy.audio.AudioClip import AudioArrayClip
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

# def requirements():
#     os.system('cmd /c "pip install mediainfo"')
#     os.system('cmd /c "pip install ffmpeg"')
#     os.system('cmd /c "pip install spleeter"')
    
# requirements

#Setting up GUI
#Window or Display Size
root = Tk()
root.wm_title("Music Assassin")
root.geometry("795x450")
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
    
class PrintLogger(object):  # create file like object
    def __init__(self, textbox):  # pass reference to text widget
        self.textbox = textbox  # keep ref
    def write(self, text):
        self.textbox.configure(state="normal")  # make field editable
        self.textbox.insert("end", text)  # write text to textbox
        self.textbox.see("end")  # scroll to end
        self.textbox.configure(state="disabled")  # make field readonly
    def flush(self):  # needed for file like object
        pass
    
def Logger():
    log_widget = ScrolledText(root, height=20, width=120, font=("consolas", "8", "normal"))
    log_widget.pack()
    logger = PrintLogger(log_widget)
    sys.stdout = logger
    sys.stderr = logger

def get_file_path():
    global browse
    # Open and return file path
    browse = filedialog.askopenfilename(title = "Select A File", filetypes = (("mp4", "*.mp4"), (".mp3","*.mp3"), (".wav","*.wav"), ("wmv", "*.wmv"), ("avi", "*.avi"), (".pcm","*.pcm"), (".aiff","*.aiff"), (".aac","*.aac"), (".ogg","*.ogg"), (".wma","*.wma"), (".flac","*.flac"), (".alac","*.alac"),("All files", "*.*")), multiple=True)

    
def demusic(file_path):
    """ testmoviepy """
    
    # Making names of files
    file_path = file_path.replace("\\", "/")
    name = (((file_path.split("/"))[-1]).split("."))[0]
    extension = (((file_path.split("/"))[-1]).split("."))[-1]
    path = "/".join(file_path.split("/")[:-1])
    output_name = name + "-no-music." + extension
    audio_extensions = ["pcm", "wav", "aiff", "mp3", "aac", "ogg", "wma", "flac", "alac"]

    # Loading audio and video
    print("Loading audio and video:")
    root.update()
    if extension not in audio_extensions:
        clip = VideoFileClip(file_path)
        waveform = clip.audio.to_soundarray()
        sample_rate = clip.audio.fps
    else:
        clip = AudioFileClip(file_path)
        waveform = clip.to_soundarray()
        sample_rate = clip.fps

    # Spliting audio into smaller parts
    print("Spliting audio into smaller parts:")
    root.update()
    t = int(sample_rate)*60 # s
    no_split = np.ceil(waveform.shape[0]/t)
    wave = np.array_split(waveform,no_split, axis=0)

    # Perform the separation :
    print("Starting Seperation")
    print("No. Sections:",len(wave))
    root.update()
    separator = Separator('spleeter:2stems')
    no_music = np.zeros(waveform.shape)
    start = 0
    for i,wav in enumerate(wave):
        print("Section:", i+1)
        root.update()
        root.update_idletasks()
        pred = separator.separate(wav)
        no_music[start:start+wav.shape[0],:] = pred['vocals']
        start += wav.shape[0]
    
    #Generating audio from numpy
    audio_no_music = AudioArrayClip(no_music, fps=sample_rate)
    
    if extension not in audio_extensions:
        # Joining audio and video
        print("Rejoining audio and video:")
        root.update()
        no_music_clip = clip.set_audio(audio_no_music)

        # Exporting
        print("Exporting video:")
        root.update()
        root.update_idletasks()
        no_music_clip.write_videofile(path+'/'+output_name)
    else:
        # Exporting
        print("Exporting audio:")
        root.update()
        audio_no_music.write_audiofile(path+'/'+output_name)

    print('Done')
    
def demusic_list():
    global files
    j = 1
    for i in files:
        demusic(i)
        j += 1
    
    #d2 = Label(root, text = "Done", wraplength=300, bg="black", fg="white", justify="left").pack(side=TOP, expand=None, fill=X)
    files = []

#Title Labels
Labels()

files = []
file_path = ""
browse = [""]
b1 = Button(text = "Browse", command = get_file_path, width = 15, bg="black", fg="white",font = "Verdana 10 bold").pack(side=TOP, expand=None, fill=X)
f1 = Button(text = "Demusic", command = demusic_list, width = 15, bg="black", fg="white",font = "Verdana 10 bold").pack(side=TOP, expand=None, fill=X)
Logger()
print("Setting up GUI...")

while 1:
    for i in browse:
        file_path = i
        if file_path not in files and file_path != "":
            #l1 = Label(root, text = "File path: \n" + str(file_path), wraplength=300, justify="left").pack(side=TOP, expand=None, fill=X)
            print(file_path)
            root.update()
            files.append(file_path)
    browse = [""]
    root.update()

print("...\n")

print('GUI Setup completed')
root.mainloop()
