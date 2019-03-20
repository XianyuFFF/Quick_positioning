import os
import shutil
from config_default import configs

video_dir = "../../Src/view-Contour2.mp4"
output_snippets_dir = "../../Src/view-Contour2"

openpose = configs['openpose']

openpose_args = dict(
                video=video_dir,
                write_json=output_snippets_dir,
                display=0,
                render_pose=0,
                # maximize_positives=True,
                model_pose='COCO',
                net_resolution="-1x480",
                model_folder=configs['openpose_model_folder'])

command_line = openpose + ' '
command_line += ' '.join(['--{} {}'.format(k, v) for k, v in openpose_args.items()])
shutil.rmtree(output_snippets_dir, ignore_errors=True)
os.makedirs(output_snippets_dir)
os.system(command_line)