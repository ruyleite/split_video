#!/usr/bin/env python3
import argparse
import datetime
import torch
import open_clip
import cv2
from sentence_transformers import util
from PIL import Image
import json
import os
from moviepy.editor import VideoFileClip
from pathlib import Path

class SplitVideo:
    def __init__(self):
        self.model = None
        self.preprocess = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        self.frame_rate = 0
        self.chunks = {'last_chunk': 0}
        self.list_split_videos = []
        self.load_model()

    def load_model(self):
        # image processing model
        print('Carregando o modelo ViT-B-16-plus-240')
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
        self.model.to(self.device)

    def imageEncoder(self, img):
        img1 = Image.fromarray(img).convert('RGB')
        img1 = self.preprocess(img1).unsqueeze(0).to(self.device)
        img1 = self.model.encode_image(img1)
        return img1

    def generateScore(self, test_img, data_img):
        img1 = self.imageEncoder(test_img)
        img2 = self.imageEncoder(data_img)
        cos_scores = util.pytorch_cos_sim(img1, img2)
        score = round(float(cos_scores[0][0])*100, 2)
        return score

    def get_video_info(self, filename):
        print(f'Lendo info do arquivo {filename}')
        self.cap = cv2.VideoCapture(filename)

        self.total_frames = int(round(self.cap.get(cv2.CAP_PROP_FRAME_COUNT), 2))
        print('Total de Frames:', self.total_frames)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'Width/Height: {self.width}x{self.height} ')

        self.fps = int(round(self.cap.get(cv2.CAP_PROP_FPS), 0))
        print('FPS:', self.fps)
        self.frame_rate = 1 / self.fps
        print('Frame Rate:', self.frame_rate)

    def get_time_frame(self, frame):
        return round(frame * self.frame_rate, 0)

    def check_score(self, start, end, step, threshold, level=0):
        image_last = None
        image_before = None
        image_actual = None

        frame_before = start
        frame_end = start

        for frame in range(start, end + 1, step):
            image_last = image_before
            image_before = image_actual

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            self.cap.grab()
            success, image_actual = self.cap.retrieve()

            if not success:
                break

            if frame == start:
                print(' ' * level, '--> chunks:', self.chunks['last_chunk'], ', frame:', frame, ', start:', start,
                      ', end:', end, ' step:', step, ', threshold:', threshold, ', score: 100, frame_before:',
                      frame_before, ', frame_end:', frame_end)
                continue

            score = self.generateScore(image_actual, image_before)

            frame_aux = frame_end
            frame_end = frame

            print(' ' * level, '--> chunks:', self.chunks['last_chunk'], ', frame:', frame, ', start:', start, ', end:',
                  end, ' step:', step, ', threshold:', threshold, ', score: ', score,
                  ', frame_before:', frame_before, ', frame_end:', frame_end)

            if score <= threshold:
                if not image_last is None:
                    score_aux = self.generateScore(image_actual, image_last)

                    if score_aux > threshold:
                        self.chunks[self.chunks['last_chunk']]['frame_end'] = frame_end
                        if not frame in self.chunks[self.chunks['last_chunk']]['frame']:
                            self.chunks[self.chunks['last_chunk']]['frame'][frame] = {}
                        self.chunks[self.chunks['last_chunk']]['frame'][frame]['score'] = score
                        continue

                print('=' * level, '===> encontrando ponto que quebra de imagem')
                print(' ' * level, f'     frame: {frame}, score: {score}')
                if step > 1:
                    self.check_score(start=frame_aux, end=frame, step=1, threshold=threshold, level=4)
                    frame_before = self.chunks[self.chunks['last_chunk']]['frame_before']
                else:
                    if len(self.chunks[self.chunks['last_chunk']]['frame']) > 5:
                        frame_end = frame_before
                        frame_before = frame
                        self.chunks['last_chunk'] += 1
                        print(' ' * level, 'Criando o chunk', self.chunks['last_chunk'])
                        print(' ' * level, '--> chunks:', self.chunks['last_chunk'], ', frame:', frame, ', start:', start,
                              ', end:', end, ' step:', step, ', threshold:', threshold, ', score: ', score,
                              ', frame_before:', frame_before, ', frame_end:', frame_end)

                        self.chunks[self.chunks['last_chunk']] = {'frame_before': frame_before, 'frame_end': frame_end,
                                                                  'frame': {frame: {'score': score}}}
                        continue

            self.chunks[self.chunks['last_chunk']]['frame_end'] = frame_end
            if not frame in self.chunks[self.chunks['last_chunk']]['frame']:
                self.chunks[self.chunks['last_chunk']]['frame'][frame] = {}
            self.chunks[self.chunks['last_chunk']]['frame'][frame]['score'] = score

    def split_videos(self, full_video):
        print('*' * 120)
        print('*', 'Gravando os Clips baseados no chunks'.center(116) , '*')
        print('*' * 120)
        filename, extension = os.path.splitext(full_video)

        clip = VideoFileClip(full_video)
        current_duration = clip.duration

        self.frame_rate = current_duration / self.total_frames

        for chunk in self.chunks:
            if chunk == "last_chunk":
                continue

            if (self.chunks[chunk]['frame_end'] - self.chunks[chunk]['frame_before']) < 5:
                continue

            start = self.get_time_frame(self.chunks[chunk]['frame_before'])
            end = self.get_time_frame(self.chunks[chunk]['frame_end'])

            if (end - start) < 1:
                continue

            current_video = f"{filename}_{str(chunk).zfill(4)}{extension}"
            print('-->Gravando o clip:', current_video, start, end)
            clip = VideoFileClip(full_video, verbose=True)
            subclip = clip.subclip(start, end)
            subclip.to_videofile(current_video,
                                 codec="libx264",
                                 temp_audiofile='temp-audio.m4a',
                                 remove_temp=True,
                                 audio_codec='aac')
            subclip.close()
            self.chunks[chunk]['filename'] = current_video
            self.list_split_videos.append(current_video)
            print('='*120)

    def analytic_video(self, full_video, start, end, step, threshold):
        print('*' * 120)
        print('*', 'Analizando o video e dividindo em chunks'.center(116), '*')
        print('*' * 120)

        self.chunks = {'last_chunk': start}

        print('Criando o chunk', self.chunks['last_chunk'])

        self.chunks[self.chunks['last_chunk']] = {'frame_before': start, 'frame_end': 0,
                                                  'frame': {start: {'score': 100}}}

        self.check_score(start, end, step, threshold)

        filename, extension = os.path.splitext(full_video)
        with open(f'{filename}.json', 'w') as f:
            json.dump(self.chunks, f, indent=4)

    def split(self, filename, step=0, threshold=90.0, clip_only=False):
        full_video = filename

        time_start = datetime.datetime.now()
        self.get_video_info(full_video)

        if step == 0:
            step = self.fps

        start = 0
        end = self.total_frames

        if not clip_only:
            self.analytic_video(full_video, start, end, step, threshold)

            time_stop = datetime.datetime.now()
            print('='*120)
            print('Hora de inicio de analise:', time_start)
            print('Hora de fim de analise:', time_stop)
            print('Tempo de analise:', time_stop - time_start)
            print('=' * 120)

        time_start = datetime.datetime.now()
        filename, extension = os.path.splitext(full_video)

        if clip_only:
            js = Path(f'{filename}.json')
            if js.is_file():
                with open(f'{filename}.json', 'r') as f:
                    self.chunks = json.load(f)

        self.split_videos(full_video)

        time_stop = datetime.datetime.now()
        print('='*120)
        print('Hora de inicio de criação de clips:', time_start)
        print('Hora de fim de criação de clips:', time_stop)
        print('Tempo de criação de clips:', time_stop - time_start)
        print('=' * 120)

        with open(f'{filename}.json', 'w') as f:
            json.dump(self.chunks, f, indent=4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--filename", required=True)
    ap.add_argument("-s", "--step", required=False, default=0)
    ap.add_argument("-t", "--threshold", required=False, default=90)
    ap.add_argument("-c", "--clip_only", action='store_true', required=False, default=False)
    args = vars(ap.parse_args())

    splitVideo = SplitVideo()
    splitVideo.split(filename=args['filename'],
                     step=int(args['step']),
                     threshold=float(args['threshold']),
                     clip_only=bool(args['clip_only']))
