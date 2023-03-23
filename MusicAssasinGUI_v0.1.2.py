""" GUI - With logger - FFMPEG with pipes - Snipping ending audios - Closes without error - Uses lesser Memory - Has Progressbar """

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
from tkinter.ttk import Progressbar
# from tqdm.tk import trange, tqdm
# from tqdm.auto import tqdm

# For Demusic
from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator
import numpy as np
import ffmpeg

def Labels(): #Label Functions
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
    global root
    global separator
    
    global process1
    global process2
    global process3
    global sample_rate
                                             
    # Making names of files
    file_path = file_path.replace("\\", "/")
    name = (((file_path.split("/"))[-1]).split("."))[0]
    extension = (((file_path.split("/"))[-1]).split("."))[-1]
    path = "/".join(file_path.split("/")[:-1])
    output_path = path + '/' + name + "-no-music." + extension
    audio_extensions = ["pcm", "wav", "aiff", "mp3", "aac", "ogg", "wma", "flac", "alac"]

    # Separating audio and video by ffmpeg
    print("-> Separating audio and video")
    root.update()
    input_ = ffmpeg.input(file_path,  noaccurate_seek=None)
    if extension not in audio_extensions:
        video = input_.video
    else:
        video = None
    
    setup_pipes(input_,video,extension,output_path)
    out,_ = process1.communicate(input=input_)
    process1.kill()
    
    waveform = np.frombuffer(out, dtype='<f4') # Buffer to Numpy
    waveform_shape = waveform.shape
    waveform = waveform[:,np.newaxis] 
    t = sample_rate*30 # number of samples
    no_split = int(np.ceil(waveform.shape[0]/t))
    
    # Perform the separation :
    print("-> Starting Seperation")
    root.update()
    no_music = np.zeros(waveform_shape)
    overlap = sample_rate*3
    start = 0
    
    pbar = Progressbar(root, orient=HORIZONTAL, length=200, mode='determinate')
    pbar.pack(side=TOP, expand=None, fill=X)
    s = 100//no_split
    for i in range(no_split):
        root.update()
        if i == 0:
            pred = separator.separate(waveform[start:start+t+overlap])
            no_music[start:start+t]= np.swapaxes(pred['vocals'],0,1)[0][0:t]
    
        else:
            pred = separator.separate(waveform[start-overlap:start+t+overlap])
            no_music[start:start+t]= np.swapaxes(pred['vocals'],0,1)[0][overlap:t+overlap]
            
        start += t
        pbar.step(s)
        root.update_idletasks()
    pbar.pack_forget()

    
    print("-> Exporting")
    root.update()
    no_music = no_music.astype('<f4').tobytes() # Numpy to Buffer
    err = process2.communicate(input=no_music)
    process2.kill()
        
    print("Done")
    process1 = 0
    process2 = 0
    os.system("echo 'q' >stop")
    
#     del locals("demusic")
    
def demusic_list():
    global files
    global browse
    
    j = len(files)
    for i in range(j):
        print(f"File {i+1}/{j}")
        demusic(files[i])
    files = []
    browse = [""]
    print("\nReady to demusic...")
        
def deAudio_list():
    global files
    global browse
    j = len(files)
    for i in range(j):
        print(f"File {i+1}/{j}")
        deAudio(files[i])
    files = []
    browse = [""]
    print("\nReady to demusic...")
    
def deAudio(file_path):
    
    # Making names of files
    print("-> Deciding Names")
    root.update()
    file_path = file_path.replace("\\", "/")
    name = (((file_path.split("/"))[-1]).split("."))[0]
    extension = (((file_path.split("/"))[-1]).split("."))[-1]
    path = "/".join(file_path.split("/")[:-1])
    no_audio_path = path + '/' + name + "-no-audio." + extension
    
    print("-> Importing File")
    root.update()
    input_ = ffmpeg.input(file_path)
    print("-> Removing Audio")
    root.update()
    input_video = input_.video
    
    print("-> Exporting File")
    root.update()
    input_video.output(no_audio_path).run()
    
    print("Done")


def setup_pipes(input_,video,extension,output_path):
    global process1
    global process2
    global process3
    global sample_rate       
#     pipe:0 is for stdin and pipe:1 is for stdout
    process1 = (
        input_
        .output('pipe:1', format='f32le', ac=1, ar=sample_rate)
        .overwrite_output()
        .run_async(pipe_stdout=True, pipe_stderr=True, quiet=True))
    
    if video == None:
        process2 = (
        ffmpeg
            .input('pipe:0', format='f32le', ar=sample_rate)
            .output(output_path, format=extension, strict='experimental')
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stderr=True, quiet=True))
    else:
        process2 = (
            ffmpeg
            .input('pipe:0', format='f32le', ar=sample_rate)
            .output(video, filename=output_path, format=extension,  vcodec='copy', acodec='aac', strict='experimental')
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stderr=True, quiet=True))
    
def on_closing():
    global browse
    global process1
    global process
    if browse == [""]:
        if process1:
            process1.kill()
        if process2:
            process2.kill()
        root.destroy()
    else:
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            if process1:
                process1.kill()
            if process2:
                process2.kill()
            root.destroy()

            
#Setting up GUI
root = Tk()
root.wm_title("Music Assassin")
root.geometry("400x500") #Window or Display Size
root.configure(background = 'white')
root.protocol("WM_DELETE_WINDOW", on_closing)

Labels() #Title Labels
files = []
file_path = ""
browse = [""]
b1 = Button(text = "Browse", command = get_file_path, width = 15, bg="black", fg="white",font = "Verdana 10 bold").pack(side=TOP, expand=None, fill=X)
f1 = Button(text = "Demusic", command = demusic_list, width = 15, bg="black", fg="white",font = "Verdana 10 bold").pack(side=TOP, expand=None, fill=X)
a1 = Button(text = "Remove Audio", command = deAudio_list, width = 15, bg="black", fg="white",font = "Verdana 10 bold").pack(side=TOP, expand=None, fill=X)
Logger()
print("Ready to demusic...")

process1 = 0
process2 = 0
separator = Separator('spleeter:2stems') 
sample_rate = 44100      

if __name__ == '__main__':
    for i in browse:
        file_path = i
        if file_path not in files and file_path != "":
            print(file_path)
            root.update()
            files.append(file_path)
    browse = [""]
    root.update()
    
root.mainloop()
