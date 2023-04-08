# Music-Assassin
Removes music from long audio and video files.

## Download Desktop app
- *Latest* v.0.2.0  https://github.com/mAb-Engineers/Music-Assassin/releases/download/v0.2.0/MusicAssassin_Setup.exe
- Older    v.0.1.0 https://github.com/mAb-Engineers/Music-Assassin/releases/download/v0.1.0/Setup.exe
![image](https://user-images.githubusercontent.com/110847037/230733812-03edbc6c-6d24-4def-a3aa-8cd9255cadc2.png)![image](https://user-images.githubusercontent.com/110847037/230733834-bb4d8aa4-c7ae-43b3-924c-49ca1a3209a9.png)



*This is still in the testing phase so there may be errors. Please give feedback in such a case.  
Your feedback will be much appreciated.*

## Versions History

### Version 0.0.1
- Uses Moviepy to import video or audio by functions VideoFileClip() or AudioFileClip()
- Uses tosoundarray() function of moviepy to convert sound track to numpy array
- Splits numpy array into 1 minute sections. As spleeter has a max limit of minutes or size of file it can process depending on your RAM. 1 minute was chosen as it does not 
- .separate() function of spleeter is used and just the filterd vocals are saved in a new array
- AudioArrayClip() fuction of moviepy is used to convert the (filtered vocals) numpy array back to audio
- .write_videofile() or .write_audiofile() is used export the filtered video/audio

-- The filtering process is very slow and progress of process cannot be observed 

### Version 0.0.2
- Same working as previous version
- Has a Logger() that displays the progress of the process

-- Moviepy is very slow in exporting the video/audio
-- Moviepy uses FFMPEG at backend but is much slower than it

### Version 0.1.0
- Uses FFMPEG instead of moviepy to import video or audio by function ffmpeg.input()
- Uses .output('pipe:1', format='f32le', ac=1, ar=sample_rate) of ffmpeg to convert sound track to buffer. The output is piped and saved in a variable.
- Uses np.frombuffer(out, dtype='<f4') function of numpy to convert buffer to numpy array
- The filtering of the numpy array by spleeter is done by the same method as in the previous versions
- astype('<f4').tobytes() fuction of numpy is used to convert the (filtered vocals) numpy array to byte array
- ffmpeg.input('pipe:0', format='f32le', ar=sample_rate) function of ffmpeg is used to convert byte array to sound track
- .output() of ffmpeg is used join the video and audio, and then export it.

-- Spawning of GUI when seperator() of spleeter is used in the Desktop App
-- When GUI is closed it shows error in the Desktop App

### Version 0.1.1
- Similar working as previous version 
- Extra samples added at both sides of each 1 minute section before sending to spleeter and removed later to increase accuracy of filtering
- Added Remove Audio option that uses ffmpeg 
- Resolved error when GUI is closed by using root.protocol("WM_DELETE_WINDOW", on_closing) of Tkinter
- Used for loop to make it possible to select many files at the same time

### Version 0.1.2
- Similar working as previous version 
- Uses Progressbar() of tkinter.ttk to show progress of process
- Uses lesser memory as code cleaned
- Compiled ffmpeg processes into a single function to reduce cluter
- Kills ffmpeg processes on closing

-- Spawning error not resolved yet

### Version 0.2.1
- Similar working as previous version 
- Code rearranged to decrease startup time
- Queue is properly maintained and videos can be added during processing

### Version 0.2.0
- Similar working as previous version 
- Spawning error partially resolved using solution at https://github.com/deezer/spleeter/issues/725
- Uses lesser memory as code cleaned

-- Spawns in only first loop and it closes them itself


## Message to fellow Programmers
We want to make a real-time music filter, which may be in the form of a app that controls the devices sound controls, or as an addon for open source player like VLC.
You can use the code to develop this further, or help us in producing it. The version history with existing errors are documented above.
