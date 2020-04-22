import subprocess

subprocess.call("python main.py --input ./input-f \
--video_root ./videos \
--output ./sample.json \
--model ./resnext-101-kinetics.pth \
--resnet_shortcut B \
--model_name resnext  \
--model_depth 101 \
--mode feature \
--overlapping 1 \
--sample_duration 11 \
--save_folder testF", shell = True)
