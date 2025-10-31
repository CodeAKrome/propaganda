#!/usr/bin/env python
"""
News Article to Video Generator – 100 % local, free, GPU-accelerated.
Creates narrated news clips with:
  - TTS via Kokoro (Metal/CUDA)
  - Stable-Diffusion backgrounds tuned to *concrete* entities (Poseidon, Eurofighter, 101st Airborne…)
  - Procedural portraits, banners, transitions
  - Optional --data file (ukraine.txt) for object-level visuals
  - FIXED: "too many open files" – audio clips are loaded into RAM and closed immediately
No API keys, no cloud, Mac VideoToolbox hw-encode ready.
"""

import argparse
import json
import hashlib
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from diffusers import AutoPipelineForText2Image
from kokoro import KPipeline
from moviepy.editor import (
    AudioFileClip, CompositeVideoClip, ImageClip, TextClip,
    VideoClip, concatenate_audioclips, concatenate_videoclips
)
from moviepy.audio.AudioClip import AudioArrayClip
from PIL import Image, ImageDraw, ImageFont


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def parse_report_file(path: str) -> list:
    """Parse a .reporter file to extract relationship tuples."""
    text = Path(path).read_text(encoding="utf-8")
    m = re.search(r"<relations>(.*?)</relations>", text, flags=re.S)
    if not m:
        return []
    
    block = m.group(1)
    # Regex to find tuples of strings: ("source", "target", "relation", "description")
    pattern = r'\("([^"]+)",\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\)'
    relationships = re.findall(pattern, block)
    return relationships

def parse_entity_file(path: str) -> dict:
    """Convert ukraine.txt <entities> block into dict."""
    text = Path(path).read_text(encoding="utf-8")
    m = re.search(r"<entities>(.*?)</entities>", text, flags=re.S)
    if not m:
        sys.exit("ERROR: --data file lacks <entities> … </entities> block")
    block = m.group(1)
    out = {k: [] for k in
           "CARDINAL DATE EVENT FAC GPE LAW LOC MONEY NORP ORDINAL ORG PERSON PRODUCT QUANTITY TIME WORK_OF_ART LANGUAGE PERCENT".split()}
    for line in block.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()].extend([x.strip() for x in v.split(",")])
    return out


def load_audio_as_array(path: Path, sr: int = 24_000) -> np.ndarray:
    """Load entire audio into RAM, close file-handle immediately."""
    with AudioFileClip(str(path)) as clip:
        arr = clip.to_soundarray(fps=sr)  # (N,2) stereo
        if arr.ndim == 2 and arr.shape[1] == 2:
            arr = arr.mean(axis=1)        # mono
    return arr

def _find_relevant_relationships(self, sent: str, relationships: list) -> list:
    """Find relationships where entities are mentioned in the sentence."""
    relevant = []
    sent_lower = sent.lower()
    for rel in relationships:
        source, target, _, _ = rel
        # Check if either the source or target entity is in the sentence
        if re.search(rf'\b{re.escape(source)}\b', sent, re.I) or \
            re.search(rf'\b{re.escape(target)}\b', sent, re.I):
            relevant.append(rel)
    return relevant

# ------------------------------------------------------------------
# Core generator
# ------------------------------------------------------------------
class NewsVideoGenerator:
    def __init__(self, voice="af_heart", resolution=(1920, 1080)):
        self._setup_metal()
        self.pipeline = KPipeline(lang_code="a")
        self._setup_img_pipe()
        self.voice = voice
        self.resolution = resolution
        self.sample_rate = 24_000
        self.cache_dir = Path("/tmp/video_cache")
        self.cache_dir.mkdir(exist_ok=True)

    # ---------- GPU ----------
    def _setup_metal(self):
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("✓ Metal GPU acceleration enabled", file=sys.stderr)
            torch.set_default_device("mps")
            torch.set_default_dtype(torch.float32)
            torch.backends.mps.enable_mps_fallback = True
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("✓ CUDA GPU acceleration enabled", file=sys.stderr)
        else:
            self.device = torch.device("cpu")
            print("⚠ Using CPU", file=sys.stderr)

    # ---------- SD Pipeline ----------
    def _setup_img_pipe(self):
        print("Initializing SD pipeline …", file=sys.stderr)
        tmp = torch.get_default_device()
        torch.set_default_device("cpu")  # avoid MPS scheduler bug
        self.image_pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo")
        torch.set_default_device(tmp)

    # ---------- Entity-aware background ----------
    def _entities_in_sentence(self, sent: str, txt_entities: dict) -> dict:
        found = {k: [] for k in txt_entities}
        for ent_type, surf_list in txt_entities.items():
            for surf in surf_list:
                if re.search(rf'\b{re.escape(surf)}\b', sent, flags=re.I):
                    found[ent_type].append(surf)
        return found

    def get_background_for_segment(self, segment: dict, duration: float,
                                   txt_entities: dict = None, relationships: list = None) -> ImageClip:
        if not txt_entities:  # fallback to generic if no entity data
            return self.create_ai_background(segment, duration)

        entities = self._entities_in_sentence(segment['text'], txt_entities)
        prompt_parts = []
        if entities["PRODUCT"]:
            prompt_parts.extend(entities["PRODUCT"])
        if entities["EVENT"]:
            prompt_parts.extend(entities["EVENT"])
        if entities["FAC"]:
            prompt_parts.extend(entities["FAC"])
        if entities["ORG"] and "NATO" in " ".join(entities["ORG"]):
            prompt_parts.append("NATO base")
        if entities["CARDINAL"]:
            prompt_parts.append(f"{entities['CARDINAL'][0]} units")

        # If still no specific parts, fall back to generic topics
        if not prompt_parts:
            prompt_parts = segment['topics'] + segment['places']
            
        prompt = ", ".join(prompt_parts)
        full_prompt = (f"cinematic digital art, news broadcast background, {prompt}. "
                       f"mood: {segment['sentiment']}, highly detailed, 4K")

        prompt_hash = hashlib.md5(full_prompt.encode()).hexdigest()
        cached = self.cache_dir / f"bg_{prompt_hash}.png"
        if cached.exists():
            img = Image.open(cached)
        else:
            print(f"🎨 Generating object-level background: {prompt}", file=sys.stderr)
            img = self.image_pipeline(prompt=full_prompt,
                                      num_inference_steps=2,
                                      guidance_scale=0.0).images[0]
            img.save(cached)

        return ImageClip(np.array(img.resize(self.resolution))).set_duration(duration)

    # ---------- Legacy generic background ----------
    def create_ai_background(self, segment: dict, duration: float) -> ImageClip:
        prompt_parts = segment["topics"] + segment["places"]
        prompt = ", ".join(prompt_parts)
        full_prompt = (f"cinematic digital art, news background, {prompt}, "
                       f"mood: {segment['sentiment']}, high detail")
        prompt_hash = hashlib.md5(full_prompt.encode()).hexdigest()
        cached = self.cache_dir / f"bg_{prompt_hash}.png"
        if cached.exists():
            img = Image.open(cached)
        else:
            print(f"Generating AI background: {prompt}", file=sys.stderr)
            img = self.image_pipeline(prompt=full_prompt,
                                      num_inference_steps=2,
                                      guidance_scale=0.0).images[0]
            img.save(cached)
        return ImageClip(np.array(img.resize(self.resolution))).set_duration(duration)

    # ---------- TTS ----------
    def generate_audio(self, text: str, output_path: Path) -> float:
        print(f"TTS: {text[:60]}…", file=sys.stderr)
        if self.device.type in ("mps", "cuda"):
            try:
                with torch.device(self.device):
                    self.pipeline.model.to(self.device)
                    generator = self.pipeline(text, voice=self.voice)
                    chunks = [audio.cpu() for gs, ps, audio in generator]
            except Exception as e:
                print(f"⚠ GPU TTS failed, fallback CPU: {e}", file=sys.stderr)
                generator = self.pipeline(text, voice=self.voice)
                chunks = [audio for gs, ps, audio in generator]
        else:
            generator = self.pipeline(text, voice=self.voice)
            chunks = [audio for gs, ps, audio in generator]

        valid = [c for c in chunks if c is not None and c.numel() > 0]
        if not valid:
            print("⚠ No audio generated, creating 0.1 s silence", file=sys.stderr)
            valid = [torch.zeros(int(0.1 * self.sample_rate), dtype=torch.float32)]
        audio = torch.cat(valid).cpu()
        sf.write(output_path, audio, self.sample_rate)

        # safety file-size check
        if not output_path.exists() or output_path.stat().st_size < 1024:
            print("⚠ Audio file empty, overwriting with 0.1 s silence", file=sys.stderr)
            silent_audio = np.zeros(int(0.1 * self.sample_rate), dtype=np.float32)
            sf.write(output_path, silent_audio, self.sample_rate)
            return 0.1

        return len(audio) / self.sample_rate

    # ---------- Text parsing ----------
    def parse_article(self, text: str):
        text = re.sub(r'[^\w\s.,!?;:\'"()\-]', '', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        segments = []
        for sent in sentences:
            if not sent.strip():
                continue
            segments.append({
                "text": sent.strip(),
                "people": self._extract_people(sent),
                "places": self._extract_places(sent),
                "topics": self._extract_topics(sent),
                "sentiment": self._analyze_sentiment(sent)
            })
        return segments

    def _extract_people(self, text: str):
        pat = r'\b(?:President|Governor|Mayor|Mr\.|Ms\.|Dr\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        matches = re.findall(pat, text)
        matches += re.findall(r'\b(?:by|from|with)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', text)
        return list(set(matches))

    def _extract_places(self, text: str):
        known = ["Chicago", "Memphis", "America", "United States", "Tennessee",
                 "Illinois", "Washington", "New York", "Los Angeles", "San Francisco",
                 "Russia", "Ukraine", "Moscow", "Ankara", "Germany", "Turkey"]
        return [p for p in known if p in text]

    def _extract_topics(self, text: str):
        kw = {"crime": ["crime", "enforcement"], "politics": ["Trump", "administration"],
              "immigration": ["immigration"], "military": ["National Guard", "troops"],
              "city": ["city", "Mayor"]}
        topics = []
        low = text.lower()
        for t, ws in kw.items():
            if any(w in low for w in ws):
                topics.append(t)
        return topics

    def _analyze_sentiment(self, text: str):
        pos, neg = ["approval", "support"], ["rejected", "war zones"]
        low = text.lower()
        cpos = sum(low.count(w) for w in pos)
        cneg = sum(low.count(w) for w in neg)
        return "negative" if cneg > cpos else "positive" if cpos > cneg else "neutral"

    # ---- overlays ------------------------------------------------------------
    def create_person_overlay(self, name, duration, position='right'):
        """Create an overlay with person's portrait and name."""
        portrait = self.create_placeholder_portrait(name, style='professional')
        portrait = portrait.resize((300, 300))
        
        overlay_width = 350
        overlay_height = 400
        overlay_img = Image.new('RGBA', (overlay_width, overlay_height), (0, 0, 0, 0))
        
        # Semi-transparent background with gradient
        for y in range(overlay_height):
            alpha = int(200 - (y / overlay_height) * 50)
            bg = Image.new('RGBA', (overlay_width, 1), (20, 40, 80, alpha))
            overlay_img.paste(bg, (0, y))
        
        # Paste portrait
        portrait_rgba = portrait.convert('RGBA')
        overlay_img.paste(portrait_rgba, (25, 20), portrait_rgba)
        
        # Add name text with glow
        draw = ImageDraw.Draw(overlay_img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 28)
        except:
            try:
                font = ImageFont.truetype("Arial Bold", 28)
            except:
                font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), name, font=font)
        text_width = bbox[2] - bbox[0]
        x = (overlay_width - text_width) // 2
        
        # Glow effect
        for offset in range(3, 0, -1):
            alpha = 80 - offset * 20
            draw.text((x+offset, 342+offset), name, fill=(0, 0, 0, alpha), font=font)
        
        # Main text
        draw.text((x, 340), name, fill=(255, 255, 255, 255), font=font)
        
        clip = ImageClip(np.array(overlay_img)).set_duration(duration)
        
        if position == 'right':
            clip = clip.set_position((self.resolution[0] - overlay_width - 50, 150))
        else:
            clip = clip.set_position((50, 150))
        
        clip = clip.crossfadein(0.3).crossfadeout(0.3)
        
        return clip

    def create_placeholder_portrait(self, name, style='professional'):
        """Create a high-quality procedural portrait with multiple styles."""
        img = Image.new('RGB', (400, 400), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Generate consistent colors from name
        hash_val = int(hashlib.md5(name.encode()).hexdigest(), 16)
        hue = (hash_val % 360)
        
        # Convert HSV to RGB for background
        h, s, v = hue / 360.0, 0.6, 0.4
        c = v * s
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = v - c
        
        if h < 1/6:
            r, g, b = c, x, 0
        elif h < 2/6:
            r, g, b = x, c, 0
        elif h < 3/6:
            r, g, b = 0, c, x
        elif h < 4/6:
            r, g, b = 0, x, c
        elif h < 5/6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        bg_color = (int((r+m)*255), int((g+m)*255), int((b+m)*255))
        
        # Create gradient background
        for y in range(400):
            ratio = y / 400
            darker = tuple(int(c * (0.7 + 0.3 * ratio)) for c in bg_color)
            draw.line([(0, y), (400, y)], fill=darker)
        
        if style == 'professional':
            # Draw circle background for avatar
            circle_color = tuple(min(255, int(c * 1.3)) for c in bg_color)
            draw.ellipse([50, 50, 350, 350], fill=circle_color)
            
            # Add subtle border
            draw.ellipse([50, 50, 350, 350], outline=(255, 255, 255, 128), width=3)
            
            # Get initials
            parts = name.split()
            initials = ''.join([p[0] for p in parts[:2]]).upper()
            
            # Draw initials with shadow
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 120)
            except:
                try:
                    font = ImageFont.truetype("Arial Bold", 120)
                except:
                    font = ImageFont.load_default()
            
            # Text shadow
            bbox = draw.textbbox((0, 0), initials, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (400 - text_width) // 2
            y = (400 - text_height) // 2 - 10
            
            # Shadow
            draw.text((x+3, y+3), initials, fill=(0, 0, 0, 180), font=font)
            # Main text
            draw.text((x, y), initials, fill='white', font=font)
            
        elif style == 'silhouette':
            # Create a silhouette-style portrait
            # Head shape
            draw.ellipse([100, 80, 300, 280], fill=(40, 40, 50))
            # Shoulders
            draw.polygon([(120, 280), (280, 280), (350, 400), (50, 400)], fill=(40, 40, 50))
            # Add rim light effect
            draw.arc([100, 80, 300, 280], 200, 340, fill=(200, 200, 255), width=3)
        
        return img

    def create_text_overlay(self, segment: dict, duration: float):
        w, h = self.resolution
        overlays = []
        if segment["places"]:
            loc = TextClip(segment["places"][0], fontsize=45, color="white",
                           font="Arial-Bold", bg_color="rgba(200,0,0,0.9)",
                           method="caption").set_position((30, 30)).set_duration(duration)
            overlays.append(loc)
        subtitle = TextClip(segment["text"], fontsize=36, color="white",
                            font="Arial", bg_color="rgba(0,0,0,0.85)",
                            size=(w - 200, None), method="caption", align="center"
                            ).set_position(("center", h - 180)).set_duration(duration)
        overlays.append(subtitle.crossfadein(0.2))
        return overlays

    def create_news_banner(self, duration: float, ticker: str = "BREAKING NEWS"):
        w, _ = self.resolution

        def make(t):
            img = Image.new("RGB", (w, 90), (180, 0, 0))
            draw = ImageDraw.Draw(img)
            try:
                f1 = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 38)
                f2 = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 24)
            except:
                f1 = f2 = ImageFont.load_default()
            draw.text((30, 25), ticker, fill="white", font=f1)
            offset = int(t * 100) % w
            txt = "● LIVE COVERAGE ● LATEST UPDATES ● " * 10
            draw.text((w - offset, 60), txt, fill="yellow", font=f2)
            return np.array(img)

        return VideoClip(make, duration=duration).set_position(("center", 0))

    # ---------- Final composer ----------
    def generate_video(self, article_text: str, output_video_path: str,
                       txt_entities: dict = None, relationships: list = None) -> None:
        print("Parsing article …", file=sys.stderr)
        segments = self.parse_article(article_text)
        print(f"Found {len(segments)} segments", file=sys.stderr)

        video_clips, audio_clips = [], []
        t = 0

        # ---------- hardware-encoding block (unchanged) --------------------
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
        # -------------------------------------------------------------------

        for i, seg in enumerate(segments):
            print(f"\n=== Segment {i+1}/{len(segments)} ===", file=sys.stderr)
            audio_path = self.cache_dir / f"segment_{i}.wav"
            
            try:
                # Generate audio first
                self.generate_audio(seg['text'], audio_path)
                # Then, get its *exact* duration for perfect sync
                with AudioFileClip(str(audio_path)) as ac:
                    dur = ac.duration
                if dur < 0.1:
                    print(f"⚠ Skipping segment {i+1} – audio too short ({dur:.2f}s)", file=sys.stderr)
                    continue
            except Exception as e:
                print(f"⚠ Error generating audio for segment {i+1}, skipping: {e}", file=sys.stderr)
                continue

            # Now that we have the exact duration, build the video segment
            try:
                bg = self.get_background_for_segment(seg, dur, txt_entities)
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

                # Load audio and add to list
                audio_clip = AudioFileClip(str(audio_path)).set_start(t)
                audio_clips.append(audio_clip)

                t += dur
            except Exception as e:
                print(f"⚠ Error creating video segment {i+1}, skipping: {e}", file=sys.stderr)
                continue

        print("\n=== Compositing final video ===", file=sys.stderr)

        if not video_clips:
            sys.exit("No video clips were generated. Exiting.")

        final_video = CompositeVideoClip(video_clips)
        final_audio = concatenate_audioclips(audio_clips)
        final_video.audio = final_audio

        print(f"Writing video to {output_video_path}...", file=sys.stderr)
        final_video.write_videofile(
            output_video_path, fps=30, codec='libx264', audio_codec='aac',
            bitrate='8000k', temp_audiofile=str(self.cache_dir / 'temp_audio.m4a'),
            remove_temp=True, ffmpeg_params=ffmpeg_params, threads=8, logger=None
        )
        final_video.close()
        print("\n✓ Video generation complete!", file=sys.stderr)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Local news-video generator")
    parser.add_argument("input", help="Article text file or '-' for stdin")
    parser.add_argument("output", help="Output .mp4")
    parser.add_argument("--data", metavar="ENTITIES.txt",
                        help="Newswire metadata (<entities> block) for object-level backgrounds")
    parser.add_argument("--report", metavar="REPORT.reporter",
                        help="Report file (<relations> block) for relationship-based backgrounds")    
    args = parser.parse_args()

    print("=== System ===", file=sys.stderr)
    print(f"Platform: {platform.system()} {platform.machine()}", file=sys.stderr)
    print(f"PyTorch: {torch.__version__}", file=sys.stderr)
    if platform.system() == "Darwin":
        print(f"Metal: {torch.backends.mps.is_available()}", file=sys.stderr)

    # read article
    if args.input == "-":
        text = sys.stdin.read()
    else:
        text = Path(args.input).read_text(encoding="utf-8")

    # parse optional entity file
    txt_entities = parse_entity_file(args.data) if args.data else None
    relationships = parse_report_file(args.report) if args.report else None
    
    # render
    t0 = time.time()
    gen = NewsVideoGenerator()
    gen.generate_video(text, args.output, txt_entities, relationships)
    print(f"\n⚡ Total: {time.time() - t0:.1f}s", file=sys.stderr)
    print(f"✓ Saved: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()