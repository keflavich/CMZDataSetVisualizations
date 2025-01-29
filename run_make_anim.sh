#!/bin/bash
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=adamginsburg@ufl.edu     # Where to send mail
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --mem=64gb                     # Job memory request
#SBATCH --time=96:00:00               # Time limit hrs:min:sec
#SBATCH --output=/red/adamginsburg/ACES/logs/ACES_animations_%j.log
#SBATCH --export=ALL
#SBATCH --job-name=ACES_animations
#SBATCH --qos=astronomy-dept-b
#SBATCH --account=astronomy-dept
pwd; hostname; date


source /orange/adamginsburg/miniconda3/bin/activate /orange/adamginsburg/miniconda3/envs/python312


/blue/adamginsburg/adamginsburg/miniconda3/envs/python312/bin/python /orange/adamginsburg/cmz/DataSetVisualizations/cmz_zoom_animation.py
convert zoom_anim_cmz_linear_HI-CO-Dust_m0.35.gif \( -clone 0 -set delay 500 \) \( -clone 1-179 \) -delete 0-179 \( +clone -set delay 500 \) +swap +delete \( -clone 1--1 -reverse \) -loop 0 zoom_anim_cmz_linear_HI-CO-Dust_m0.35_withpause.gif
ffmpeg -y -i zoom_anim_cmz_linear_HI-CO-Dust_m0.35_withpause.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" zoom_anim_cmz_linear_HI-CO-Dust_m0.35_withpause.mp4


/blue/adamginsburg/adamginsburg/miniconda3/envs/python312/bin/python /orange/adamginsburg/cmz/DataSetVisualizations/ACES_Animation.py
#fr=40; delay=200; convert zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_segment7.gif \( -clone 0-$fr \) \( -clone $fr -set delay $delay \) \( -clone $fr-60 \) -delete 0-61 zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_segment7_withpause.gif
fr=40; delay=200; convert zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_segment7.gif \( -clone $fr -set delay $delay \) -swap $fr,-1 +delete zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_segment7_withpause.gif
#seg=6; fr=30; delay=200; convert zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_segment${seg}.gif \( -clone 0-$fr \) \( -clone $fr -set delay $delay \) \( -clone $fr-60 \) -delete 0-61 zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_segment${seg}_withpause.gif
seg=6; fr=30; delay=200; convert zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_segment${seg}.gif \( -clone $fr -set delay $delay \) -swap $fr,-1 +delete zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_segment${seg}_withpause.gif
#seg=8; fr=18; delay=200; convert zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_segment${seg}.gif \( -clone 0-$fr \) \( -clone $fr -set delay $delay \) \( -clone $fr-60 \) -delete 0-61 zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_segment${seg}_withpause.gif
seg=8; fr=18; delay=200; convert zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_segment${seg}.gif \( -clone $fr -set delay $delay \) -swap $fr,-1 +delete zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_segment${seg}_withpause.gif
convert zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_segment[0-3].gif \
        zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_segment4.gif \
        \( +clone -set delay 200 \) \
        zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_segment5.gif \
        \( +clone -set delay 200 \) \
        zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_segment6_withpause.gif \
        zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_segment7_withpause.gif \
        zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_segment8_withpause.gif \
        zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_segment9.gif \
        \( +clone -set delay 500 \) +swap +delete \
        \( -clone 0--1 -reverse \) -loop 0 \
        zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_combined.gif
ffmpeg -y -i zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_combined.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_combined.mp4

# wrong order... ls zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_segment*mp4 > mylist.txt
ffmpeg -y -f concat -i mylist.txt -c copy zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_combined_frommp4.mp4
# use reverse order because the frame number changes as you go
ffmpeg -y -i zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_combined_frommp4.mp4 \
        -vf "loop=125:1:780,\
            loop=100:1:635,\
            loop=50:1:540,\
            loop=50:1:430,\
            loop=50:1:400,\
            loop=75:1:293,\
            loop=75:1:230,\
            setpts=N/FRAME_RATE/TB" zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_combined_frommp4_withpause.mp4
# just for debugging ffmpeg -y -i zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_combined_frommp4.mp4 -vf loop=75:1:230,setpts=N/FRAME_RATE/TB,loop=75:1:300,setpts=N/FRAME_RATE/TB zoom_anim_cmz_linear_withACES_HI-CO-Dust_m0.35_combined_frommp4_withpause.mp4

# https://superuser.com/questions/1071369/ffmpeg-pause-video-every-10-seconds-for-3-seconds

/blue/adamginsburg/adamginsburg/miniconda3/envs/python312/bin/python /orange/adamginsburg/cmz/DataSetVisualizations/SgrB2_Zoom_Animation.py
#convert cmz_to_sgrb2_zoomier_square_segment[0-9].gif \( -clone 0 -set delay 200 \) \( -clone 1-239 \) \( +clone -set delay 200 \) \( -clone 241-419 \) -delete 0-419  \( +clone -set delay 500 \) +swap +delete \( -clone 1--1 -reverse \) -loop 0 cmz_to_sgrb2_zoomier_combined_square.gif
#ffmpeg -y -i cmz_to_sgrb2_zoomier_combined_square.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" cmz_to_sgrb2_zoomier_combined_square.mp4
ffmpeg -y -f concat -i cmz_to_sgrb2_zoomier_square_segment.list -c copy cmz_to_sgrb2_zoomier_square_combined_frommp4.mp4
ffmpeg -y -i cmz_to_sgrb2_zoomier_square_combined_frommp4.mp4 -vf "loop=500:1:419,loop=200:1:240,loop=100:1:0,setpts=N/FRAME_RATE/TB" cmz_to_sgrb2_zoomier_square_combined_frommp4_withpause.mp4

/blue/adamginsburg/adamginsburg/miniconda3/envs/python312/bin/python /orange/adamginsburg/cmz/DataSetVisualizations/ACES_pan_Animation.py
ffmpeg -y -i pan_anim_ACES.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" pan_anim_ACES2.mp4
