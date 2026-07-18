""" GUI - With logger - FFMPEG - Snipping ending audios - Closes without error """

# For GUI
import time
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import os
#from tqdm import tqdm
import warnings

# For Logger
import sys
from tkinter import Tk, Button, Frame
from tkinter.scrolledtext import ScrolledText

# For Demusic
from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator
import numpy as np
import ffmpeg

# warnings.simplefilter("ignore")

# def requirements():
#     os.system('cmd /c "pip install mediainfo"')
#     os.system('cmd /c "pip install ffmpeg"')
#     os.system('cmd /c "pip install spleeter"')
    
# requirements

#Setting up GUI
#Window or Display Size

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
    for i in browse:
        file_path = i
        if file_path not in files and file_path != "":
            print(file_path)
            root.update()
            files.append(file_path)
    
    
def demusic(file_path): 
    """ testffmpeg """
    sample_rate = 44100
    # Making names of files
    file_path = file_path.replace("\\", "/")
    name = (((file_path.split("/"))[-1]).split("."))[0]
    extension = (((file_path.split("/"))[-1]).split("."))[-1]
    path = "/".join(file_path.split("/")[:-1])
    output_path = path + '/' + name + "-no-end-30s-music." + extension
    audio_extensions = ["pcm", "wav", "aiff", "mp3", "aac", "ogg", "wma", "flac", "alac"]

    # Separating audio and video by ffmpeg
    print("Separating audio and video:")
    root.update()
    input_ = ffmpeg.input(file_path,  noaccurate_seek=None)
    if extension not in audio_extensions:
        video = input_.video
#     out,_ = input_.output('pipe:1', fo  rmat='f32le', ac=1, ar=sample_rate).overwrite_output().run(capture_stdout=True, capture_stderr=True)
    process1 = (
        input_
            .output('pipe:1', format='f32le', ac=1, ar=sample_rate)
            .overwrite_output()
            .run_async(pipe_stdout=True, pipe_stderr=True, quiet=True))

    out,_ = process1.communicate()
    process1.kill()
    process1.wait()
    
    # Buffer to Numpy
    print("Converting audio:")
    root.update()
    waveform = np.frombuffer(out, dtype='<f4')
    waveform1 = np.hstack((waveform[:,np.newaxis],waveform[:,np.newaxis]))
    # waveform1 = np.zeros((waveform.shape[0],2))
    # waveform1[:,0] = waveform

    t = sample_rate*30 # number of samples
    no_split = int(np.ceil(waveform1.shape[0]/t))
#     wave = np.array_split(waveform1,no_split, axis=0)

    # Perform the separation :
    print("Starting Seperation")
    print("No. Sections:",no_split)
    root.update()
    separator = Separator('spleeter:2stems')
    no_music = np.zeros(waveform.shape)
    overlap = sample_rate*3
    start = 0
    for i in range(no_split):
        print("Section:",i+1)
        root.update()
        if i == 0:
            pred = separator.separate(waveform1[start:start+t+overlap])
            no_music[start:start+t]= np.swapaxes(pred['vocals'],0,1)[0][0:t]
            
#         elif i+1 >= no_split:
#             pred = separator.separate(waveform1[start-10:])
#             no_music[start:]= np.swapaxes(pred['vocals'][10:t+10],0,1)[0]
        else:
            pred = separator.separate(waveform1[start-overlap:start+t+overlap])
            no_music[start:start+t]= np.swapaxes(pred['vocals'],0,1)[0][overlap:t+overlap]
            
        start += t

    # Numpy to Buffer
    print("Converting audio:")
    root.update()
    no_music = no_music.astype('<f4').tobytes()

    if extension not in audio_extensions:
        print("Exporting video:")
        root.update()
        process = (
            ffmpeg
            .input('pipe:0', format='f32le', ar=sample_rate)
            .output(video, output_path, format=extension,  vcodec='copy', acodec='aac', strict='experimental')
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stderr=True, quiet=True))

    else:
        print("Exporting audio:")
        root.update()
        process = (
            ffmpeg
            .input('pipe:0', format='f32le', ar=sample_rate)
            .output(output_path, format=extension, strict='experimental')
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stderr=True, quiet=True))

    err = process.communicate(input=no_music)
    process.kill()
    process.wait()
    print("Done")
#     del locals("demusic")
    
def demusic_list():
    global files
    j = 1
    for i in files:
        demusic(i)
        j += 1    
    #d2 = Label(root, text = "Done", wraplength=300, bg="black", fg="white", justify="left").pack(side=TOP, expand=None, fill=X)
    files = []
    browse = [""]
        
def deAudio_list():
    global files
    j = 1
    for i in files:
        deAudio(i)
        j += 1    
    #d2 = Label(root, text = "Done", wraplength=300, bg="black", fg="white", justify="left").pack(side=TOP, expand=None, fill=X)
    files = []
    browse = [""]
    
def deAudio(file_path):
    
    # Making names of files
    print("Deciding Names:")
    root.update()
    file_path = file_path.replace("\\", "/")
    name = (((file_path.split("/"))[-1]).split("."))[0]
    extension = (((file_path.split("/"))[-1]).split("."))[-1]
    path = "/".join(file_path.split("/")[:-1])
    no_audio_path = path + '/' + name + "-no-audio." + extension
    
    print("Importing File:")
    root.update()
    input_ = ffmpeg.input(file_path)
    print("Removing Audio:")
    root.update()
    input_video = input_.video
    
    print("Exporting File:")
    root.update()
    input_video.output(no_audio_path).run()
    
    print("Done")

def on_closing():
    global browse 
    global x 
    if browse != [""]:
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            x = 0
            root.destroy()
    else:
        x = 0
        root.destroy()

root = Tk()
root.wm_title("Music Assassin")
root.geometry("400x500")
root.configure(background = 'white')
root.protocol("WM_DELETE_WINDOW", on_closing)

#Title Labels
Labels()
files = []
file_path = ""
browse = [""]
b1 = Button(text = "Browse", command = get_file_path, width = 15, bg="black", fg="white",font = "Verdana 10 bold").pack(side=TOP, expand=None, fill=X)
f1 = Button(text = "Demusic", command = demusic_list, width = 15, bg="black", fg="white",font = "Verdana 10 bold").pack(side=TOP, expand=None, fill=X)
a1 = Button(text = "Remove Audio", command = deAudio_list, width = 15, bg="black", fg="white",font = "Verdana 10 bold").pack(side=TOP, expand=None, fill=X)
Logger()
print("Ready to demusic...")

if __name__ == '__main__':
    x = 0
    while x:
        for i in browse:
            file_path = i
            if file_path not in files and file_path != "":
                #l1 = Label(root, text = "File path: \n" + str(file_path), wraplength=300, justify="left").pack(side=TOP, expand=None, fill=X)
                print(file_path)
                root.update()
                files.append(file_path)
        browse = [""]
        root.update()
    x = 1
root.mainloop()
