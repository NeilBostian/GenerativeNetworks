import os

import cv2

def stitch_image_files_to_video(
        images,
        output_file_path = 'output.avi',
        fourcc='DIVX', # One of http://www.fourcc.org/codecs.php
        fps=120,
        frame_size=(1920, 1080),
        isColor=True
    ):
    vwriter = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*fourcc), fps, frame_size, isColor)

    images = list(images)
    complete = 0

    for image_name in images:
        img = cv2.imread(image_name)
        vwriter.write(img)

        complete += 1
        print(f'{complete}/{len(images)} frames complete')

    vwriter.release()

def main():
    def get_image_paths():
        in_dir = 'F:\\repos\\ml\\fractals\\.data\\legacy\\imgs'
        for i in range(0, 600):
            yield f'{in_dir}/f-{i}.png'

    stitch_image_files_to_video(get_image_paths(), )

if __name__ == '__main__':
    main()