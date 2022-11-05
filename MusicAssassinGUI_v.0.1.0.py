""" GUI - With logger - FFMPEG """

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
from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator
import numpy as np
import ffmpeg

# def requirements():
#     os.system('cmd /c "pip install mediainfo"')
#     os.system('cmd /c "pip install ffmpeg"')
#     os.system('cmd /c "pip install spleeter"')
    
# requirements

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
    """ testffmpeg """
    sample_rate = 44100
    # Making names of files
    file_path = file_path.replace("\\", "/")
    name = (((file_path.split("/"))[-1]).split("."))[0]
    extension = (((file_path.split("/"))[-1]).split("."))[-1]
    path = "/".join(file_path.split("/")[:-1])
    output_path = path + '/' + name + "-no-music." + extension
    audio_extensions = ["pcm", "wav", "aiff", "mp3", "aac", "ogg", "wma", "flac", "alac"]


    # Separating audio and video by ffmpeg
    print("Separating audio and video:")
    root.update()
    input_ = ffmpeg.input(file_path,  noaccurate_seek=None)
    video = input_.video
    out,_ = input_.output('pipe:', format='f32le', ac=1, ar=sample_rate).overwrite_output().run(capture_stdout=True, capture_stderr=True)

    # Buffer to Numpy
    print("Converting audio:")
    root.update()
    waveform = np.frombuffer(out, dtype='<f4')
    waveform1 = np.hstack((waveform[:,np.newaxis],waveform[:,np.newaxis]))
    # waveform1 = np.zeros((waveform.shape[0],2))
    # waveform1[:,0] = waveform

    t = sample_rate*60 # number of samples
    no_split = np.ceil(waveform1.shape[0]/t)
    wave = np.array_split(waveform1,no_split, axis=0)

    # Perform the separation :
    print("Starting Seperation")
    print("No. Sections:",len(wave))
    root.update()
    separator = Separator('spleeter:2stems')
    no_music = np.zeros(waveform.shape)
    start = 0
    for i,wav in enumerate(wave):
        print("Section:",i+1)
        root.update()
        root.update_idletasks()
        pred = separator.separate(wav)
        no_music[start:start+wav.shape[0]]= np.swapaxes(pred['vocals'],0,1)[0]
    #     no_music[i*wav.shape[0]:(i+1)*wav.shape[0]]= np.swapaxes(pred['vocals'],0,1)[0]
        start += wav.shape[0]

    # Numpy to Buffer
    print("Converting audio:")
    root.update()
    no_music = no_music.astype('<f4').tobytes()

    if extension not in audio_extensions:
        print("Exporting video:")
        root.update()
        root.update_idletasks()
        process = (
            ffmpeg
            .input('pipe:', format='f32le', ar=sample_rate)
            .output(video, output_path, format=extension,  vcodec='copy', acodec='aac', strict='experimental')
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stderr=True, quiet=True))
    else:
        print("Exporting audio:")
        root.update()
        root.update_idletasks()
        process = (
            ffmpeg
            .input('pipe:', format='f32le', ar=sample_rate)
            .output(output_path, format=extension, strict='experimental')
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stderr=True, quiet=True))

    err = process.communicate(input=no_music)
    print("Done")
    
def demusic_list():
    global files
    j = 1
    for i in files:
        demusic(i)
        j += 1    
    #d2 = Label(root, text = "Done", wraplength=300, bg="black", fg="white", justify="left").pack(side=TOP, expand=None, fill=X)
    files = []
    
    
if __name__ == '__main__':
    #Title Labels
    Labels()

    files = []
    file_path = ""
    browse = [""]
    b1 = Button(text = "Browse", command = get_file_path, width = 15, bg="black", fg="white",font = "Verdana 10 bold").pack(side=TOP, expand=None, fill=X)
    f1 = Button(text = "Demusic", command = demusic_list, width = 15, bg="black", fg="white",font = "Verdana 10 bold").pack(side=TOP, expand=None, fill=X)
    Logger()
    print("Ready to demusic...")

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

