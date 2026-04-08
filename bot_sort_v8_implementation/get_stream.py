import yt_dlp

URL = "https://www.earthcam.com/usa/connecticut/newlondon/?cam=newlondon" # Panama Canal

ydl_opts = {'format': 'best'}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(URL, download=False)
    
    print("\n" + "="*50)
    print("SUCCESS! PASTE THIS URL INTO YOUR SCRIPT:")
    print("="*50)
    print(info['url'])
    print("="*50 + "\n")
