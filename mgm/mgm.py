#!/usr/bin/env python
"""
News Article to Video Generator – Fully Local & Free
Creates a narrated MP4 from plain text:  procedural visuals,
AI-generated portraits, TTS audio, animated overlays.
No API keys or paid services required.
Optimised for Apple Silicon (Metal) and VideoToolbox hardware encoding.
"""

import sys
import re
import json
import hashlib
import platform
import tempfile
import shutil
from pathlib import Path
from kokoro import KPipeline
from diffusers import AutoPipelineForText2Image
import soundfile as sf
import torch
from moviepy.editor import (
    VideoClip, ImageClip, TextClip, CompositeVideoClip,
    AudioFileClip, concatenate_videoclips, concatenate_audioclips
)
from moviepy.audio.AudioClip import AudioArrayClip
from diffusers import DPMSolverMultistepScheduler
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from io import BytesIO
import time
import math

# ---------- generator --------------------------------------------------------
class NewsVideoGenerator:
    def __init__(self, voice='af_heart', resolution=(1920, 1080)):
        self._setup_metal_acceleration()
        self.pipeline = KPipeline(lang_code='a')
        self._setup_image_pipeline()
        self.voice = voice
        self.resolution = resolution
        self.sample_rate = 24_000
        self.cache_dir = Path('/tmp/video_cache')
        self.cache_dir.mkdir(exist_ok=True)

    # ---- device setup -------------------------------------------------------
    def _setup_metal_acceleration(self):
        if platform.system() == 'Darwin':                      # macOS
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
                print("✓ Metal GPU acceleration enabled", file=sys.stderr)
                torch.set_default_device('mps')
                torch.set_default_dtype(torch.float32)
                torch.backends.mps.enable_mps_fallback = True
            else:
                self.device = torch.device('cpu')
                print("⚠ Metal GPU not available, using CPU", file=sys.stderr)
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print("✓ CUDA GPU acceleration enabled", file=sys.stderr)
            else:
                self.device = torch.device('cpu')
                print("⚠ Using CPU", file=sys.stderr)

    # ---- diffusion pipeline -------------------------------------------------
    def _setup_image_pipeline(self):
        print("Initializing AI image generation pipeline...", file=sys.stderr)
        original = torch.get_default_device()
        torch.set_default_device('cpu')          # avoid MPS tensors in scheduler
        try:
            self.image_pipeline = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sd-turbo"
            )
        finally:
            torch.set_default_device(original)

    # ---- procedural portraits ----------------------------------------------
    def create_placeholder_portrait(self, name, style='professional'):
        img = Image.new('RGB', (400, 400), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        # consistent colours from name hash
        h = int(hashlib.md5(name.encode()).hexdigest(), 16) % 360
        s, v = 0.6, 0.4
        c = v * s; x = c * (1 - abs((h / 60.0) % 2 - 1)); m = v - c
        if h < 60:   r, g, b = c, x, 0
        elif h < 120: r, g, b = x, c, 0
        elif h < 180: r, g, b = 0, c, x
        elif h < 240: r, g, b = 0, x, c
        elif h < 300: r, g, b = x, 0, c
        else:         r, g, b = c, 0, x
        bg = tuple(int((ch + m) * 255) for ch in (r, g, b))

        # gradient background
        for y in range(400):
            alpha = y / 400
            shade = tuple(int(c * (0.7 + 0.3 * alpha)) for c in bg)
            draw.line([(0, y), (400, y)], fill=shade)

        if style == 'professional':
            circle = tuple(min(255, int(c * 1.3)) for c in bg)
            draw.ellipse([50, 50, 350, 350], fill=circle, outline=(255, 255, 255), width=3)
            initials = ''.join(w[0] for w in name.split()[:2]).upper()
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 120)
            except:
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), initials, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x, y = (400 - tw) // 2, (400 - th) // 2 - 10
            draw.text((x + 3, y + 3), initials, fill=(0, 0, 0, 180), font=font)
            draw.text((x, y), initials, fill='white', font=font)
        return img

    # ---- procedural backgrounds -------------------------------------------
    def create_procedural_cityscape(self, city_name, duration, time_of_day='night'):
        w, h = self.resolution
        seed = int(hashlib.md5(city_name.encode()).hexdigest(), 16) % 10_000
        np.random.seed(seed)
        n_build = 20
        builds = []
        for i in range(n_build):
            x = (w // n_build) * i
            bw = w // n_build + np.random.randint(-20, 40)
            bh = np.random.randint(200, 600)
            builds.append((x, bw, bh))

        top = (10, 15, 40) if time_of_day == 'night' else (100, 150, 200)
        bot = (30, 35, 60) if time_of_day == 'night' else (150, 180, 220)
        img = Image.new('RGB', (w, h))
        draw = ImageDraw.Draw(img)
        for y in range(h):
            ratio = y / h
            r = int(top[0] + (bot[0] - top[0]) * ratio)
            g = int(top[1] + (bot[1] - top[1]) * ratio)
            b = int(top[2] + (bot[2] - top[2]) * ratio)
            draw.line([(0, y), (w, y)], fill=(r, g, b))

        for i, (x, bw, bh) in enumerate(builds):
            y = h - bh
            col = (20 + i % 30,) * 3 if time_of_day == 'night' else (100 + i % 50,) * 3
            draw.rectangle([x, y, x + bw, h], fill=col)
            if time_of_day == 'night':
                rows, cols = bh // 40, bw // 30
                for row in range(rows):
                    for col in range(cols):
                        wx, wy = x + 10 + col * 30, y + 20 + row * 40
                        if (seed + i + row + col) % 20 < 15:
                            bright = 200 + int(55 * math.sin(i + row + col))
                            draw.rectangle([wx, wy, wx + 15, wy + 20],
                                         fill=(bright, bright, 150))
        return ImageClip(np.array(img)).set_duration(duration)

    def create_government_building(self, duration):
        w, h = self.resolution
        img = Image.new('RGB', (w, h), (40, 50, 80))
        draw = ImageDraw.Draw(img)
        bw, bh = 800, 600
        bx = (w - bw) // 2; by = h - bh
        draw.rectangle([bx, by, bx + bw, h], fill=(180, 180, 190))
        for i in range(8):
            cx = bx + (i + 1) * (bw // 9)
            shadow = int(20 + 10 * math.sin(i))
            draw.rectangle([cx - 20, by + 100, cx + 20, h],
                         fill=(200 - shadow,) * 3)
        draw.polygon([(bx - 50, by + 100), (bx + bw + 50, by + 100),
                    (bx + bw // 2, by - 50)], fill=(160, 160, 170))
        return ImageClip(np.array(img)).set_duration(duration)

    def create_police_scene(self, duration):
        w, h = self.resolution
        img = Image.new('RGB', (w, h), (20, 25, 40))
        draw = ImageDraw.Draw(img)
        for rad in range(100, 0, -10):
            draw.ellipse([200 - rad, 200 - rad, 200 + rad, 200 + rad],
                       fill=(200, 0, 50))
            draw.ellipse([w - 200 - rad, 200 - rad, w - 200 + rad, 200 + rad],
                       fill=(50, 50, 200))
        return ImageClip(np.array(img)).set_duration(duration)

    def create_military_scene(self, duration):
        w, h = self.resolution
        img = Image.new('RGB', (w, h), (60, 70, 50))
        draw = ImageDraw.Draw(img)
        np.random.seed(42)
        for i in range(50):
            x = np.random.randint(0, w); y = np.random.randint(0, h)
            size = np.random.randint(100, 300)
            cols = [(80, 90, 60), (60, 70, 50), (100, 110, 80), (50, 60, 40)]
            draw.ellipse([x, y, x + size, y + size], fill=cols[i % 4])
        return ImageClip(np.array(img)).set_duration(duration)

    def create_abstract_background(self, text, topic, duration):
        w, h = self.resolution
        schemes = {'default': [(30, 40, 60), (50, 60, 90)],
                   'news': [(40, 40, 70), (60, 60, 100)],
                   'politics': [(50, 30, 30), (80, 50, 50)]}
        top, bot = schemes.get(topic, schemes['default'])
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2 ** 32 - 1)
        np.random.seed(seed)
        img = Image.new('RGB', (w, h), top)
        draw = ImageDraw.Draw(img)
        for y in range(h):
            ratio = y / h
            r = int(top[0] * (1 - ratio) + bot[0] * ratio)
            g = int(top[1] * (1 - ratio) + bot[1] * ratio)
            b = int(top[2] * (1 - ratio) + bot[2] * ratio)
            draw.line([(0, y), (w, y)], fill=(r, g, b))
        return ImageClip(np.array(img)).set_duration(duration)

    # ---- AI background via SD-Turbo -----------------------------------------
    def create_ai_background(self, segment, duration):
        prompt_parts = segment['topics'] + segment['places']
        if segment['people']:
            prompt_parts.append(f"featuring {segment['people'][0]}")
        prompt = ", ".join(prompt_parts)
        full_prompt = (f"cinematic digital art, news broadcast background, {prompt}. "
                       f"mood: {segment['sentiment']}, high detail, sharp focus")
        phash = hashlib.md5(full_prompt.encode()).hexdigest()
        cached = self.cache_dir / f"bg_{phash}.png"
        if cached.exists():
            img = Image.open(cached)
        else:
            print(f"Generating AI background for: {prompt}", file=sys.stderr)
            img = self.image_pipeline(prompt=full_prompt, num_inference_steps=2, guidance_scale=0.0).images[0]
            img.save(cached)
        return ImageClip(np.array(img.resize(self.resolution))).set_duration(duration)

    # ---- text analysis ------------------------------------------------------
    def parse_article(self, text):
        text = re.sub(r'[^\w\s.,!?;:\'"()\-]', '', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        segments = []
        for sent in sentences:
            if not sent.strip():
                continue
            segments.append({
                'text': sent.strip(),
                'people': self._extract_people(sent),
                'places': self._extract_places(sent),
                'topics': self._extract_topics(sent),
                'sentiment': self._analyze_sentiment(sent)
            })
        return segments

    def _extract_people(self, text):
        pat = r'\b(?:President|Governor|Mayor|Mr\.|Ms\.|Dr\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        matches = re.findall(pat, text)
        matches += re.findall(r'\b(?:by|from|with)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', text)
        return list(set(matches))

    def _extract_places(self, text):
        known = ['Chicago', 'Memphis', 'America', 'United States', 'Tennessee',
                 'Illinois', 'Washington', 'New York', 'Los Angeles', 'San Francisco']
        return [p for p in known if p in text]

    def _extract_topics(self, text):
        kw = {'crime': ['crime', 'enforcement', 'operations', 'law'],
              'politics': ['Trump', 'administration', 'federal', 'government', 'election'],
              'immigration': ['immigration'],
              'military': ['National Guard', 'deployment', 'troops'],
              'city': ['city', 'cities', 'urban', 'Mayor']}
        topics = []
        txt = text.lower()
        for t, words in kw.items():
            if any(w in txt for w in words):
                topics.append(t)
        return topics

    def _analyze_sentiment(self, text):
        pos = ['approval', 'cooperative', 'support', 'success']
        neg = ['rejected', 'crime-infested', 'war zones', 'lack']
        txt = text.lower()
        pos_c = sum(txt.count(w) for w in pos)
        neg_c = sum(txt.count(w) for w in neg)
        return 'negative' if neg_c > pos_c else 'positive' if pos_c > neg_c else 'neutral'

    # ---- background selector ------------------------------------------------
    def get_background_for_segment(self, segment, duration):
        return self.create_ai_background(segment, duration)

    # ---- TTS with GPU fallback ----------------------------------------------
    def generate_audio(self, text, output_path):
        print(f"Generating audio: {text[:60]}...", file=sys.stderr)
        if hasattr(self, 'device') and self.device.type in {'mps', 'cuda'}:
            try:
                with torch.device(self.device):
                    self.pipeline.model.to(self.device)
                    gen = self.pipeline(text, voice=self.voice)
                    chunks = [audio.cpu() for gs, ps, audio in gen]
            except Exception as e:
                print(f"⚠ GPU generation failed, falling back to CPU: {e}", file=sys.stderr)
                gen = self.pipeline(text, voice=self.voice)
                chunks = [audio for gs, ps, audio in gen]
        else:
            gen = self.pipeline(text, voice=self.voice)
            chunks = [audio for gs, ps, audio in gen]

        valid = [c for c in chunks if c is not None and c.numel() > 0]
        if not valid:
            print("⚠ No audio generated, creating 0.1 s silence", file=sys.stderr)
            valid = [torch.zeros(int(0.1 * self.sample_rate), dtype=torch.float32)]
        audio = torch.cat(valid).cpu()
        sf.write(output_path, audio, self.sample_rate)

        # safety file-size check
        if not output_path.exists() or output_path.stat().st_size < 1024:
            print("⚠ Audio file empty, overwriting with 0.1 s silence", file=sys.stderr)
            sf.write(output_path, np.zeros(int(0.1 * self.sample_rate), dtype=np.float32), self.sample_rate)
        return len(audio) / self.sample_rate

    # ---- overlays ------------------------------------------------------------
    def create_person_overlay(self, name, duration, position='right'):
        portrait = self.create_placeholder_portrait(name).resize((300, 300))
        ow, oh = 350, 400
        base = Image.new('RGBA', (ow, oh), (0, 0, 0, 0))
        for y in range(oh):
            alpha = int(200 - (y / oh) * 50)
            base.paste(Image.new('RGBA', (ow, 1), (20, 40, 80, alpha)), (0, y))
        base.paste(portrait.convert('RGBA'), (25, 20))
        draw = ImageDraw.Draw(base)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 28)
        except:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), name, font=font)
        tw = bbox[2] - bbox[0]
        x = (ow - tw) // 2
        for off in range(3, 0, -1):
            draw.text((x + off, 342 + off), name, fill=(0, 0, 0, 80 - off * 20), font=font)
        draw.text((x, 340), name, fill=(255, 255, 255), font=font)
        clip = ImageClip(np.array(base)).set_duration(duration)
        x_pos = self.resolution[0] - ow - 50 if position == 'right' else 50
        return clip.set_position((x_pos, 150)).crossfadein(0.3).crossfadeout(0.3)

    def create_text_overlay(self, segment, duration):
        w, h = self.resolution
        overlays = []
        if segment['places']:
            loc = TextClip(segment['places'][0], fontsize=45, color='white',
                         font='Arial-Bold', bg_color='rgba(200,0,0,0.9)')
            overlays.append(loc.set_position((30, 30)).set_duration(duration))
        subtitle = TextClip(segment['text'], fontsize=36, color='white',
                          font='Arial', bg_color='rgba(0,0,0,0.85)',
                          size=(w - 200, None), method='caption', align='center')
        overlays.append(subtitle.set_position(('center', h - 180))
                        .set_duration(duration).crossfadein(0.2))
        return overlays

    def create_news_banner(self, duration, ticker_text="BREAKING NEWS ANALYSIS"):
        w, h = self.resolution
        def make_frame(t):
            img = Image.new('RGB', (w, 90), (180, 0, 0))
            draw = ImageDraw.Draw(img)
            try:
                f1 = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 38)
                f2 = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 24)
            except:
                f1 = f2 = ImageFont.load_default()
            draw.text((30, 25), ticker_text, fill='white', font=f1)
            scroll = int(t * 100) % w
            ticker = "● LIVE COVERAGE ● LATEST UPDATES ● " * 10
            draw.text((w - scroll, 60), ticker, fill='yellow', font=f2)
            return np.array(img)
        return VideoClip(make_frame, duration=duration).set_position(('center', 0))

    # ---- final composer -----------------------------------------------------
    def generate_video(self, article_text, output_video_path):
        print("Parsing article...", file=sys.stderr)
        segments = self.parse_article(article_text)
        print(f"Found {len(segments)} segments", file=sys.stderr)

        video_clips, audio_clips = [], []
        t = 0

        # hardware encoding opts
        ffmpeg_params = ['-preset', 'medium', '-crf', '18']
        if platform.system() == 'Darwin':
            try:
                import subprocess, re
                encoders = subprocess.check_output(['ffmpeg', '-encoders'], text=True, timeout=2)
                if 'h264_videotoolbox' in encoders:
                    print("✓ Using VideoToolbox hardware encoding", file=sys.stderr)
                    ffmpeg_params = ['-c:v', 'h264_videotoolbox', '-b:v', '8000k',
                                   '-profile:v', 'high', '-allow_sw', '1']
            except:
                pass

        for i, seg in enumerate(segments):
            print(f"\n=== Segment {i+1}/{len(segments)} ===", file=sys.stderr)
            audio_path = self.cache_dir / f"segment_{i}.wav"
            try:
                dur = self.generate_audio(seg['text'], audio_path)
                if dur <= 0.1:
                    print(f"⚠ Skipping segment {i+1} – audio too short ({dur:.2f}s)", file=sys.stderr)
                    continue
            except Exception as e:
                print(f"⚠ Error generating audio for segment {i+1}, skipping: {e}", file=sys.stderr)
                continue

            try:
                bg = self.get_background_for_segment(seg, dur)
            except Exception as e:
                print(f"⚠ Error generating background for segment {i+1}, skipping: {e}", file=sys.stderr)
                continue

            bg = bg.fl_image(lambda fr: (fr * 0.7).astype('uint8'))
            banner = self.create_news_banner(dur)
            overlays = self.create_text_overlay(seg, dur)
            person_ov = []
            for j, person in enumerate(seg['people'][:2]):
                print(f"Creating portrait for: {person}", file=sys.stderr)
                pos = 'right' if j == 0 else 'left'
                person_ov.append(self.create_person_overlay(person, dur, pos))

            all_clips = [bg, banner] + person_ov + overlays
            video = CompositeVideoClip(all_clips, size=self.resolution).set_opacity(1)
            video = video.set_duration(dur).set_start(t)
            video_clips.append(video)

            # ---- safe audio path – no bare arrays -----------------------------
            tmp_audio = str(self.cache_dir / f'segment_{i}_tmp.wav')
            shutil.copy(audio_path, tmp_audio)
            aclip = AudioFileClip(tmp_audio).set_start(t)
            audio_clips.append(aclip)
            # ------------------------------------------------------------------
            t += dur

        print("\n=== Compositing final video ===", file=sys.stderr)
        final_audio = concatenate_audioclips(audio_clips)
        # close all readers immediately
        for ac in audio_clips:
            ac.close()

        final_video = CompositeVideoClip(video_clips, size=self.resolution)
        final_video = final_video.set_audio(final_audio)

        print(f"Writing video to {output_video_path}...", file=sys.stderr)
        final_video.write_videofile(
            output_video_path, fps=30, codec='libx264', audio_codec='aac',
            bitrate='8000k', temp_audiofile=str(self.cache_dir / 'temp_audio.m4a'),
            remove_temp=True, ffmpeg_params=ffmpeg_params, threads=8, logger=None
        )
        final_video.close()
        print("\n✓ Video generation complete!", file=sys.stderr)


# ---------- CLI --------------------------------------------------------------
def main():
    if len(sys.argv) < 3:
        print("Usage: ./news_video_gen.py <input_file|-> <output.mp4>", file=sys.stderr)
        print("\n✨ 100% Local & Free – No API Keys Required", file=sys.stderr)
        sys.exit(1)

    in_file, out_file = sys.argv[1], sys.argv[2]
    print("=== System Configuration ===", file=sys.stderr)
    print(f"Platform: {platform.system()} {platform.machine()}", file=sys.stderr)
    print(f"PyTorch: {torch.__version__}", file=sys.stderr)
    if platform.system() == 'Darwin':
        print(f"Metal available: {torch.backends.mps.is_available()}", file=sys.stderr)
    print("============================\n", file=sys.stderr)

    text = sys.stdin.read() if in_file == '-' else Path(in_file).read_text(encoding='utf-8')
    t0 = time.time()
    NewsVideoGenerator().generate_video(text, out_file)
    print(f"\n⚡ Total generation time: {time.time() - t0:.1f}s", file=sys.stderr)
    print(f"✓ Video saved to: {out_file}", file=sys.stderr)


if __name__ == '__main__':
    main()