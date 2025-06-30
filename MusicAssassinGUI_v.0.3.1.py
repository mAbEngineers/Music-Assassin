""" GUI - With logger - FFMPEG with pipes - Snipping ending audios - Closes without error -
    Uses lesser Memory - No Spawning - Maintains proper queue - Reduced imports - 
    Corrected audio output - Functions reshuffled - onclose updated - Handles subs 
    - Editable list """

# For GUI
import os # For Admin Permissions
from tkinter import Tk, Button, filedialog, messagebox, Label, Frame, Scrollbar, Listbox

# For Logger
import sys  # For Logger
from tkinter.scrolledtext import ScrolledText
    
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
    
def setFrames(root):
    global b1
    global b2
    global inprogressframe
    global listbox
    global f1
    global a1
    
    fileframe = Frame(root)
    fileframe.pack( side = "top", fill="x", padx=(20,20) )
    b1 = Button(fileframe, text = "Browse",  bg="black", fg="white",font = "Verdana 10 bold")
    b1.pack(side="left", expand=True, fill="x")
    b2 = Button(fileframe, text = "Remove",  bg="black", fg="white",font = "Verdana 10 bold")
    b2.pack(side="right", expand=True, fill="x")

    inprogressframe = Frame(root)
    inprogressframe.pack( side = "top", fill="x", padx=(20,20) )

    listboxframe = Frame(root)
    listboxframe.pack( side = "top", fill="x", padx=(20,20) , pady=(0,10) )
    scrollbar_y = Scrollbar(listboxframe, orient="vertical")
    scrollbar_x = Scrollbar(listboxframe, orient="horizontal")
    listbox = Listbox(listboxframe, yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set, selectmode="multiple", height=5)
    scrollbar_y.config(command=listbox.yview)
    scrollbar_x.config(command=listbox.xview)
    scrollbar_y.pack(side="right", fill="y")
    scrollbar_x.pack(side="bottom", fill="x")
    listbox.pack(side="left", fill="both", expand=True)

    actionframe = Frame(root)
    actionframe.pack( side = "top", fill="x", padx=(20,20) )
    f1 = Button(actionframe, text = "DeMusic", bg="black", fg="white",font = "Verdana 10 bold")
    f1.pack(side="left", anchor="nw", expand=True, fill="x")
    a1 = Button(actionframe, text = "DeAudio", bg="black", fg="white",font = "Verdana 10 bold")
    a1.pack(side="right", anchor="ne", expand=True, fill="x")
    
    log_widget = ScrolledText(root, height=20, font=("consolas", "8", "normal"))
    log_widget.pack(fill="x", padx=(20,20))
    logger = PrintLogger(log_widget)
    sys.stdout = logger
#     sys.stderr = logger
    
def setLabels(root): #Label Functions
    title = Label(root, text = "Music Assassin", font = "Verdana 18 bold", bg='white')  
    title.pack(side="top", expand=None, fill="x", padx=(10,10), pady=(20, 0)) 
    
    credits = Label(root, text = "—A project of mAb Engineers",font = "Verdana 6", bg='white')  
    credits.pack(side="top", expand=None, fill="x", padx=(10, 10), pady=(0, 0))
    
    prompt = Label(root, text = "Choose the video to de-music:",font = "Verdana 13 bold", bg='white')  
    prompt.pack(side="top", expand=None, fill="x", padx=(10, 10), pady=(7, 10))
    
    
#Setting up GUI
root = Tk()
root.wm_title("Music Assassin")
root.geometry("400x600") #Window or Display Size
root.configure(background = 'white')

files = []
process1 = 0
process2 = 0
separator = 0
pbar = 0
do = 0
sample_rate = 44100   

setLabels(root) #Title Labels
setFrames(root) #Button, Logger and file list Frames

if __name__ == '__main__':
    root.update_idletasks()
    root.update()
    
    
# For GUI
from tkinter.ttk import Progressbar

# For Demusic
from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator
from numpy import frombuffer, shape, zeros, swapaxes, newaxis, ceil
from ffmpeg import input, probe


def on_closing():
    global process1
    global process
    global separator
    global pbar
    if do == 0:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        root.destroy()
    else:
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
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

class MySeparator(Separator): # Solution from: https://github.com/deezer/spleeter/issues/725
    def close_pool(self):
        if self._pool:
            self._pool.close()
            self._pool.terminate()
    
def get_file_path():
    global files
    global listbox
    # Open and return file path
    file_path = ""
    browse = [""]
    browse = filedialog.askopenfilename(title = "Select A File", filetypes = (("mp4", "*.mp4"), (".mp3","*.mp3"), (".wav","*.wav"), ("wmv", "*.wmv"), ("avi", "*.avi"), (".pcm","*.pcm"), (".aiff","*.aiff"), (".aac","*.aac"), (".ogg","*.ogg"), (".wma","*.wma"), (".flac","*.flac"), (".alac","*.alac"),("All files", "*.*")), multiple=True)
    for i in browse:
        file_path = i
        if file_path not in files and file_path != "":
            print(file_path)
            files.append(file_path)
            listbox.insert("end", file_path)
            root.update()
    browse = [""]

def remove_item():
    global listbox
    global files
    selected_checkboxs = listbox.curselection()

    for selected_checkbox in selected_checkboxs[::-1]:
        listbox.delete(selected_checkbox)
        files.pop(selected_checkbox)

def check_streams(file_path): # Solution from: https://github.com/kkroening/ffmpeg-python/issues/204 (ingles98 commented on Mar 2, 2020)
    video = None
    audio = None
    subs  = None
    streams = probe(file_path)["streams"]
    for stream in streams:
        if stream["codec_type"] == "audio":
            audio = True
        if stream["codec_type"] == "video":
            video = True
        if stream["codec_type"] == "subtitle":
            subs = True
    return (video, audio, subs)
    
    
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

    # Separating audio and video by ffmpeg
    print("-> Loading file")
    root.update()
    input_ = input(file_path,  noaccurate_seek=None)
    (video, audio, subs) = check_streams(file_path)
    if audio == None:
        print("Audio track not detected")
        return
    if video:
        video = input_.video
    if subs:
        subs = input_['2']
    
    process1 = ( #     pipe:0 is for stdin and pipe:1 is for stdout
        input_
        .output('pipe:1', format='f32le', ac=1, ar=sample_rate)
        .overwrite_output()
        .run_async(pipe_stdin=False, pipe_stdout=True, pipe_stderr=True, quiet=False))
    out,_ = process1.communicate(input=input_)
    process1.kill()
    
    waveform = frombuffer(out, dtype='<f4') # Buffer to Numpy
    waveform_shape = waveform.shape
    waveform = waveform[:,newaxis] 
    t = sample_rate*30 # number of samples
    no_split = int(ceil(waveform.shape[0]/t))
    
    # Perform the separation :
    print("-> Seperating music")
    root.update()
    no_music = zeros(waveform_shape)
    overlap = sample_rate*3
    start = 0
    
    separator = MySeparator('spleeter:2stems') 
    pbar = Progressbar(root, orient="horizontal", length=300, mode='determinate')
    pbar.pack(side="top", expand=None, fill="x", padx=(20,20))
    s = 100/no_split
    for i in range(no_split):
        if i == 0:
            pred = separator.separate(waveform[start:start+t+overlap])
            no_music[start:start+t]= swapaxes(pred['vocals'],0,1)[0][0:t]
    
        else:
            pred = separator.separate(waveform[start-overlap:start+t+overlap])
            no_music[start:start+t]= swapaxes(pred['vocals'],0,1)[0][overlap:t+overlap]
            
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
        process2 = (
            input('pipe:0', format='f32le', ar=sample_rate)
            .output(output_path, format=extension, strict='experimental')
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stderr=True, quiet=True))
    elif subs == None:
        process2 = (
            input('pipe:0', format='f32le', ar=sample_rate)
            .output(video, filename=output_path, format=extension,  vcodec='copy', acodec='aac', strict='experimental')
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stderr=True, quiet=True))
    else:
        process2 = (
            input('pipe:0', format='f32le', ar=sample_rate)
            .output(video, subs, filename=output_path, vcodec='copy', acodec='aac')
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stderr=True, quiet=True))
        
    err = process2.communicate(input=no_music)
    process2.kill()
        
    print("Done")
    process1 = 0
    process2 = 0
    
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

def on_listbox_select(event):
    global listbox
    selected_indices = listbox.curselection()
    for index in selected_indices:
        if index == 1:  # Assuming "Item 2" at index 1 is unselectable
            listbox.selection_clear(index)
    
def demusic_list():
    global files
    global pbar
    global do
    
    if do:
        root.after(1000, demusic_list)
    else:   
        i = 0
        for file_todo in files:
            print(f"File {i+1}/{len(files)}")  
            do = 1      
            files.pop(0)
            listbox.delete(0)
            inprogress = Label( inprogressframe, text=file_todo, font = "Verdana 6 italic", bg='white', anchor="w")
            inprogress.pack(side="top", expand=None, fill="x")
            
            demusic(file_todo)
            
            inprogress.pack_forget()
            if pbar == 0:
                return
            i += 1
        do = 0
        print("\nReady to demusic...")
        
def deAudio_list():
    global files
    global pbar
    global do
    
    if do:
        root.after(1000, deAudio_list)
    else:   
        i = 0
        for file_todo in files:
            print(f"File {i+1}/{len(files)}")  
            do = 1      
            files.pop(0)
            listbox.delete(0)
            inprogress = Label( inprogressframe, text=file_todo, font = "Verdana 6 italic", bg='white', anchor="w")
            inprogress.pack(side="top", expand=None, fill="x")
            
            deAudio(file_todo)
            
            inprogress.pack_forget()
            if pbar == 0:
                return
            i += 1
        do = 0
        print("\nReady to demusic...")


root.protocol("WM_DELETE_WINDOW", on_closing)
b1.configure(command = get_file_path)
b2.configure(command = remove_item)
f1.configure(command = demusic_list)
a1.configure(command = deAudio_list)

print("Ready to demusic...")

root.mainloop()
