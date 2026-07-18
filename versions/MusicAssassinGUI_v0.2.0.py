""" GUI - With logger - FFMPEG with pipes - Snipping ending audios - Closes without error - Uses lesser Memory - No Spawning  """

# For GUI
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image

# For Logger
import sys 
from tkinter import Tk, Button, Frame
from tkinter.scrolledtext import ScrolledText
from tkinter.ttk import Progressbar

# For Demusic
from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator
import numpy as np
import ffmpeg 


class MySeparator(Separator): # Solution from: https://github.com/deezer/spleeter/issues/725
    def close_pool(self):
        if self._pool:
            self._pool.close()
            self._pool.terminate()

def Labels(): #Label Functions
    title = Label(root, text = "Music Assassin", font = "Verdana 18 bold", bg='white')  
    title.pack(side="top", expand=None, fill="x", padx=(10,10), pady=(20, 0)) 
    
    credits = Label(root, text = "—A project of mAb Engineers",font = "Verdana 6", bg='white')  
    credits.pack(side="top", expand=None, fill="x", padx=(10, 10), pady=(0, 0))
    
    prompt = Label(root, text = "Choose the video to de-music:",font = "Verdana 13 bold", bg='white')  
    prompt.pack(side="top", expand=None, fill="x", padx=(10, 10), pady=(7, 10))
    
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
    log_widget = ScrolledText(root, height=20, font=("consolas", "8", "normal"))
    log_widget.pack(fill="x", padx=(20,20))
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
    
    global process1
    global process2
    global process3
    global separator
    global pbar
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
    
    process1 = ( #     pipe:0 is for stdin and pipe:1 is for stdout
        input_
        .output('pipe:1', format='f32le', ac=1, ar=sample_rate)
        .overwrite_output()
        .run_async(pipe_stdin=False, pipe_stdout=True, pipe_stderr=True, quiet=False))
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
    
    separator = MySeparator('spleeter:2stems') 
    pbar = Progressbar(root, orient="horizontal", length=300, mode='determinate')
    pbar.pack(side="top", expand=None, fill="x", padx=(20,20))
    s = 100/no_split
    for i in range(no_split):
        if i == 0:
            pred = separator.separate(waveform[start:start+t+overlap])
            no_music[start:start+t]= np.swapaxes(pred['vocals'],0,1)[0][0:t]
    
        else:
            pred = separator.separate(waveform[start-overlap:start+t+overlap])
            no_music[start:start+t]= np.swapaxes(pred['vocals'],0,1)[0][overlap:t+overlap]
            
        start += t
        if pbar:
            pbar.step(s)
            root.update_idletasks()
            root.update()
        else:
            return 
        separator.join()
        separator.close_pool()
    pbar.pack_forget()
    separator = 0
    
    print("-> Exporting")
    root.update()
    no_music = no_music.astype('<f4').tobytes() # Numpy to Buffer
    
    if video == None:
        process2,_ = (
            ffmpeg
            .input('pipe:0', format='f32le', ar=sample_rate)
            .output(output_path, format=extension, strict='experimental')
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stdout=False, pipe_stderr=True))
    else:
        process2 = (
            ffmpeg
            .input('pipe:0', format='f32le', ar=sample_rate)
            .output(video, filename=output_path, format=extension,  vcodec='copy', acodec='aac', strict='experimental')
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stdout=False, pipe_stderr=True))
        
    err = process2.communicate(input=no_music)
    process2.kill()
        
    print("Done")
    process1 = 0
    process2 = 0
    
def demusic_list():
    global files
    global browse
    global pbar
    
    j = len(files)
    for i in range(j):
        print(f"File {i+1}/{j}")
        demusic(files[i])
        if pbar == 0:
            return
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

    
def on_closing():
    global browse
    global process1
    global process
    global separator
    global pbar
    if browse == [""]:
        if process1:
            process1.kill()
        if process2:
            process2.kill()
        if separator:  
            separator.join()
            separator.close_pool()
        if pbar:
            pbar = 0
        root.destroy()
    else:
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            if process1:
                process1.kill()
            if process2:
                process2.kill()
            if separator:  
                separator.join()
                separator.close_pool()
            if pbar:
                pbar = 0
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
b1 = Button(text = "Browse", command = get_file_path,  bg="black", fg="white",font = "Verdana 10 bold").pack(side="top", expand=None, fill="x", padx=(20,20))
f1 = Button(text = "Demusic", command = demusic_list, bg="black", fg="white",font = "Verdana 10 bold").pack(side="top", expand=None, fill="x", padx=(20,20))
a1 = Button(text = "Remove Audio", command = deAudio_list, bg="black", fg="white",font = "Verdana 10 bold").pack(side="top", expand=None, fill="x", padx=(20,20))
Logger()
print("Ready to demusic...")

process1 = 0
process2 = 0
separator = 0
pbar = 0
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
