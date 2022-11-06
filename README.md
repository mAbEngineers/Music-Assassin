## Music-Assassin
Removes music from long audio and video files.

#Version 0.0.1
- Uses Moviepy to import video or audio by functions VideoFileClip() or AudioFileClip()
- Uses tosoundarray() function of moviepy to convert sound track to numpy array
- Splits numpy array into 1 minute sections. As spleeter has a max limit of minutes or size of file it can process depending on your RAM. 1 minute was chosen as it does not 
- .separate() function of spleeter is used and just the filterd vocals are saved in a new array
- AudioArrayClip() fuction of moviepy is used to convert the (filtered vocals) numpy array back to audio
- .write_videofile() or .write_audiofile() is used export the filtered video/audio

-- The filtering process is very slow and progress of process cannot be observed 

#Version 0.0.2
- Same working as previous version
- Has a Logger() that displays the progress of the process

-- Moviepy is very slow in exporting the video/audio
-- Moviepy uses FFMPEG at backend but is much slower than it

#Version 0.1.0
- Uses FFMPEG instead of moviepy to import video or audio by function ffmpeg.input()
- Uses .output('pipe:1', format='f32le', ac=1, ar=sample_rate) of ffmpeg to convert sound track to buffer. The output is piped and saved in a variable.
- Uses np.frombuffer(out, dtype='<f4') function of numpy to convert buffer to numpy array
- The filtering of the numpy array by spleeter is done by the same method as in the previous versions
- astype('<f4').tobytes() fuction of numpy is used to convert the (filtered vocals) numpy array to byte array
- ffmpeg.input('pipe:0', format='f32le', ar=sample_rate) function of ffmpeg is used to convert byte array to sound track
- .output() of ffmpeg is used join the video and audio, and then export it.

-- Spawning of GUI when seperator() of spleeter is used in the Desktop App
-- When GUI is closed it shows error in the Desktop App
