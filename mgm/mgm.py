#!/usr/bin/env python
"""
News Article to Video Generator - Fully Local & Free
Generates synchronized video with TTS audio, procedural visuals, and animations.
No API keys or paid subscriptions required.
Optimized for Mac Metal GPU acceleration.
"""

import sys
import re
import json
import hashlib
import platform
from pathlib import Path
from kokoro import KPipeline
import soundfile as sf
import torch
from moviepy.editor import (
    VideoClip, ImageClip, TextClip, CompositeVideoClip, 
    AudioFileClip, concatenate_videoclips, concatenate_audioclips
)
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from io import BytesIO
import time
import math

class NewsVideoGenerator:
    def __init__(self, voice='af_heart', resolution=(1920, 1080)):
        # Configure Metal GPU acceleration for Mac
        self._setup_metal_acceleration()
        
        self.pipeline = KPipeline(lang_code='a')
        self.voice = voice
        self.resolution = resolution
        self.sample_rate = 24000
        self.cache_dir = Path('/tmp/video_cache')
        self.cache_dir.mkdir(exist_ok=True)
        
    def _setup_metal_acceleration(self):
        """Configure PyTorch to use Metal GPU acceleration on Mac."""
        if platform.system() == 'Darwin':  # macOS
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
                print("✓ Metal GPU acceleration enabled", file=sys.stderr)
                torch.set_default_device('mps')
                torch.set_default_dtype(torch.float32) # Ensures new tensors are created on the default device
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
    
    def create_procedural_cityscape(self, city_name, duration, time_of_day='night'):
        """Create an animated procedural cityscape."""
        width, height = self.resolution
        
        # Generate consistent cityscape from name
        hash_val = int(hashlib.md5(city_name.encode()).hexdigest(), 16)
        np.random.seed(hash_val % 10000)
        
        # Generate building parameters
        num_buildings = 20
        buildings = []
        for i in range(num_buildings):
            x = (width // num_buildings) * i
            w = width // num_buildings + np.random.randint(-20, 40)
            h = np.random.randint(200, 600)
            buildings.append((x, w, h))
        
        def make_frame(t):
            # Time-based colors
            if time_of_day == 'night':
                sky_top = (10, 15, 40)
                sky_bottom = (30, 35, 60)
            else:
                sky_top = (100, 150, 200)
                sky_bottom = (150, 180, 220)
            
            img = Image.new('RGB', (width, height))
            draw = ImageDraw.Draw(img)
            
            # Draw gradient sky
            for y in range(height):
                ratio = y / height
                r = int(sky_top[0] + (sky_bottom[0] - sky_top[0]) * ratio)
                g = int(sky_top[1] + (sky_bottom[1] - sky_top[1]) * ratio)
                b = int(sky_top[2] + (sky_bottom[2] - sky_top[2]) * ratio)
                draw.line([(0, y), (width, y)], fill=(r, g, b))
            
            # Draw buildings with animated lights
            for i, (x, w, h) in enumerate(buildings):
                y = height - h
                
                # Building color
                if time_of_day == 'night':
                    building_color = (20 + i % 30, 20 + i % 30, 25 + i % 35)
                else:
                    building_color = (100 + i % 50, 100 + i % 50, 110 + i % 50)
                
                draw.rectangle([x, y, x + w, height], fill=building_color)
                
                # Draw windows with blinking effect
                if time_of_day == 'night':
                    window_rows = h // 40
                    window_cols = w // 30
                    for row in range(window_rows):
                        for col in range(window_cols):
                            wx = x + 10 + col * 30
                            wy = y + 20 + row * 40
                            
                            # Animated window lights
                            blink = (hash_val + i + row + col + int(t * 10)) % 20
                            if blink < 15:
                                brightness = 200 + int(55 * math.sin(t * 3 + i + row + col))
                                window_color = (brightness, brightness, 150)
                                draw.rectangle([wx, wy, wx + 15, wy + 20], fill=window_color)
            
            # Draw stars if night
            if time_of_day == 'night':
                np.random.seed(42)
                for i in range(100):
                    sx = np.random.randint(0, width)
                    sy = np.random.randint(0, height // 2)
                    twinkle = math.sin(t * 5 + i) * 0.5 + 0.5
                    brightness = int(200 * twinkle)
                    draw.ellipse([sx, sy, sx+2, sy+2], fill=(brightness, brightness, brightness))
            
            return np.array(img)
        
        return VideoClip(make_frame, duration=duration)
    
    def create_procedural_scene(self, scene_type, duration):
        """Create procedural scenes for various topics."""
        width, height = self.resolution
        
        if scene_type == 'government':
            return self.create_government_building(duration)
        elif scene_type == 'crime':
            return self.create_police_scene(duration)
        elif scene_type == 'military':
            return self.create_military_scene(duration)
        else:
            return self.create_abstract_background(scene_type, duration)
    
    def create_government_building(self, duration):
        """Create an animated government building scene."""
        width, height = self.resolution
        
        def make_frame(t):
            img = Image.new('RGB', (width, height), (40, 50, 80))
            draw = ImageDraw.Draw(img)
            
            # Draw classical building with columns
            building_width = 800
            building_height = 600
            building_x = (width - building_width) // 2
            building_y = height - building_height
            
            # Building base
            draw.rectangle([building_x, building_y, building_x + building_width, height],
                         fill=(180, 180, 190))
            
            # Columns
            num_columns = 8
            col_spacing = building_width // (num_columns + 1)
            for i in range(num_columns):
                col_x = building_x + (i + 1) * col_spacing
                col_width = 40
                # Animated shadow effect
                shadow = int(20 + 10 * math.sin(t + i))
                draw.rectangle([col_x - col_width//2, building_y + 100, 
                              col_x + col_width//2, height],
                             fill=(200 - shadow, 200 - shadow, 210 - shadow))
            
            # Pediment (triangular top)
            draw.polygon([
                (building_x - 50, building_y + 100),
                (building_x + building_width + 50, building_y + 100),
                (building_x + building_width // 2, building_y - 50)
            ], fill=(160, 160, 170))
            
            # Flag animation
            flag_x = building_x + building_width // 2
            flag_y = building_y - 100
            wave = math.sin(t * 3) * 20
            
            # Flag pole
            draw.line([(flag_x, flag_y), (flag_x, building_y - 50)], 
                     fill=(100, 100, 100), width=5)
            
            # Flag (animated wave)
            flag_points = [
                (flag_x, flag_y),
                (flag_x + 100 + wave, flag_y + 10),
                (flag_x + 100 + wave, flag_y + 60),
                (flag_x, flag_y + 50)
            ]
            draw.polygon(flag_points, fill=(200, 50, 50))
            
            return np.array(img)
        
        return VideoClip(make_frame, duration=duration)
    
    def create_police_scene(self, duration):
        """Create an animated police/law enforcement scene."""
        width, height = self.resolution
        
        def make_frame(t):
            img = Image.new('RGB', (width, height), (20, 25, 40))
            draw = ImageDraw.Draw(img)
            
            # Police lights effect (alternating red and blue)
            light_flash = int(t * 4) % 2
            
            # Left light
            left_intensity = 200 if light_flash == 0 else 50
            for radius in range(100, 0, -10):
                alpha = (100 - radius) * 2
                color = (left_intensity, 0, 50)
                draw.ellipse([200 - radius, 200 - radius, 
                            200 + radius, 200 + radius],
                           fill=color)
            
            # Right light
            right_intensity = 200 if light_flash == 1 else 50
            for radius in range(100, 0, -10):
                alpha = (100 - radius) * 2
                color = (50, 50, right_intensity)
                draw.ellipse([width - 200 - radius, 200 - radius,
                            width - 200 + radius, 200 + radius],
                           fill=color)
            
            # Draw police badge symbol in center
            badge_x, badge_y = width // 2, height // 2
            badge_size = 150
            
            # Badge star
            points = []
            for i in range(10):
                angle = (i * 36 - 90) * math.pi / 180
                radius = badge_size if i % 2 == 0 else badge_size // 2
                px = badge_x + radius * math.cos(angle)
                py = badge_y + radius * math.sin(angle)
                points.append((px, py))
            
            draw.polygon(points, fill=(180, 160, 50), outline=(200, 180, 70))
            
            # Badge center
            draw.ellipse([badge_x - 50, badge_y - 50, badge_x + 50, badge_y + 50],
                        fill=(100, 100, 120))
            
            return np.array(img)
        
        return VideoClip(make_frame, duration=duration)
    
    def create_military_scene(self, duration):
        """Create military/National Guard scene."""
        width, height = self.resolution
        
        def make_frame(t):
            img = Image.new('RGB', (width, height), (60, 70, 50))
            draw = ImageDraw.Draw(img)
            
            # Camouflage pattern background
            np.random.seed(42)
            for i in range(50):
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                size = np.random.randint(100, 300)
                
                pattern_colors = [(80, 90, 60), (60, 70, 50), (100, 110, 80), (50, 60, 40)]
                color = pattern_colors[i % 4]
                
                draw.ellipse([x, y, x + size, y + size], fill=color)
            
            # Draw military vehicles (simplified)
            vehicle_y = height - 300
            for i in range(3):
                vx = 300 + i * 400 + int(t * 50 % width)
                
                # Vehicle body
                draw.rectangle([vx, vehicle_y, vx + 200, vehicle_y + 100],
                             fill=(70, 80, 50))
                
                # Wheels
                draw.ellipse([vx + 30, vehicle_y + 80, vx + 70, vehicle_y + 120],
                           fill=(40, 40, 40))
                draw.ellipse([vx + 160, vehicle_y + 80, vx + 200, vehicle_y + 120],
                           fill=(40, 40, 40))
            
            # Add stars for insignia
            for i in range(3):
                sx = width // 4 + i * (width // 4)
                sy = 100
                
                # Five-pointed star
                star_points = []
                for j in range(10):
                    angle = (j * 36 - 90) * math.pi / 180
                    radius = 40 if j % 2 == 0 else 16
                    px = sx + radius * math.cos(angle)
                    py = sy + radius * math.sin(angle)
                    star_points.append((px, py))
                
                draw.polygon(star_points, fill=(200, 200, 200))
            
            return np.array(img)
        
        return VideoClip(make_frame, duration=duration)
    
    def create_abstract_background(self, topic, duration):
        """Create abstract animated background."""
        width, height = self.resolution
        
        # Topic-based color schemes
        color_schemes = {
            'default': [(30, 40, 60), (50, 60, 90)],
            'news': [(40, 40, 70), (60, 60, 100)],
            'politics': [(50, 30, 30), (80, 50, 50)],
        }
        
        colors = color_schemes.get(topic, color_schemes['default'])
        
        def make_frame(t):
            img = Image.new('RGB', (width, height))
            draw = ImageDraw.Draw(img)
            
            # Animated gradient
            for y in range(height):
                ratio = (y / height + t * 0.1) % 1.0
                r = int(colors[0][0] + (colors[1][0] - colors[0][0]) * ratio)
                g = int(colors[0][1] + (colors[1][1] - colors[0][1]) * ratio)
                b = int(colors[0][2] + (colors[1][2] - colors[0][2]) * ratio)
                draw.line([(0, y), (width, y)], fill=(r, g, b))
            
            # Floating geometric shapes
            np.random.seed(42)
            for i in range(15):
                phase = t + i * 0.5
                x = int((width / 15 * i + math.sin(phase) * 100) % width)
                y = int((height / 3 + math.cos(phase * 1.3) * 150) % height)
                size = 30 + int(20 * math.sin(phase * 2))
                
                alpha = int(100 + 50 * math.sin(phase))
                shape_color = (255, 255, 255, alpha)
                
                if i % 3 == 0:
                    draw.ellipse([x, y, x+size, y+size], fill=(200, 200, 255, 50))
                elif i % 3 == 1:
                    draw.rectangle([x, y, x+size, y+size], fill=(200, 200, 255, 50))
                else:
                    points = [(x + size//2, y), (x + size, y + size), (x, y + size)]
                    draw.polygon(points, fill=(200, 200, 255, 50))
            
            return np.array(img)
        
        return VideoClip(make_frame, duration=duration)
    
    def parse_article(self, text):
        """Parse article into segments with metadata."""
        text = re.sub(r'[^\w\s.,!?;:\'"()\-]', '', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        segments = []
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            segment = {
                'text': sentence.strip(),
                'people': self._extract_people(sentence),
                'places': self._extract_places(sentence),
                'topics': self._extract_topics(sentence),
                'sentiment': self._analyze_sentiment(sentence)
            }
            segments.append(segment)
        
        return segments
    
    def _extract_people(self, text):
        """Extract person names from text."""
        pattern = r'\b(?:President|Governor|Mayor|Mr\.|Ms\.|Dr\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        matches = re.findall(pattern, text)
        
        pattern2 = r'\b(?:by|from|with)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)'
        matches.extend(re.findall(pattern2, text))
        
        return list(set(matches))
    
    def _extract_places(self, text):
        """Extract place names from text."""
        known_places = ['Chicago', 'Memphis', 'America', 'United States', 
                       'Tennessee', 'Illinois', 'Washington', 'New York',
                       'Los Angeles', 'San Francisco']
        places = [p for p in known_places if p in text]
        return places
    
    def _extract_topics(self, text):
        """Extract key topics from text."""
        keywords = {
            'crime': ['crime', 'enforcement', 'operations', 'law'],
            'politics': ['Trump', 'administration', 'federal', 'government', 'election'],
            'immigration': ['immigration'],
            'military': ['National Guard', 'deployment', 'troops'],
            'city': ['city', 'cities', 'urban', 'Mayor']
        }
        
        topics = []
        text_lower = text.lower()
        for topic, words in keywords.items():
            if any(word.lower() in text_lower for word in words):
                topics.append(topic)
        
        return topics
    
    def _analyze_sentiment(self, text):
        """Simple sentiment analysis."""
        positive_words = ['approval', 'cooperative', 'support', 'success']
        negative_words = ['rejected', 'crime-infested', 'war zones', 'lack']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if neg_count > pos_count:
            return 'negative'
        elif pos_count > neg_count:
            return 'positive'
        return 'neutral'
    
    def get_background_for_segment(self, segment, duration):
        """Determine and create appropriate background for segment."""
        if segment['places']:
            city = segment['places'][0]
            time_of_day = 'night' if 'crime' in segment['topics'] else 'day'
            return self.create_procedural_cityscape(city, duration, time_of_day)
        elif 'crime' in segment['topics']:
            return self.create_procedural_scene('crime', duration)
        elif 'military' in segment['topics']:
            return self.create_procedural_scene('military', duration)
        elif 'politics' in segment['topics']:
            return self.create_procedural_scene('government', duration)
        else:
            return self.create_abstract_background('news', duration)
    
    def generate_audio(self, text, output_path):
        """Generate TTS audio for text with Metal GPU acceleration."""
        print(f"Generating audio: {text[:60]}...", file=sys.stderr)
        
        if hasattr(self, 'device') and self.device.type in ['mps', 'cuda']:
            try:
                with torch.device(self.device):
                    # Explicitly move the entire pipeline model to the correct device
                    self.pipeline.model.to(self.device)

                    generator = self.pipeline(text, voice=self.voice)
                    
                    audio_chunks = []
                    for i, (gs, ps, audio) in enumerate(generator):
                        audio_cpu = audio.cpu() if audio.device.type != 'cpu' else audio
                        audio_chunks.append(audio_cpu)
            except Exception as e:
                print(f"⚠ GPU generation failed, falling back to CPU: {e}", file=sys.stderr)
                generator = self.pipeline(text, voice=self.voice)
                audio_chunks = [audio for _, _, audio in generator]
        else:
            generator = self.pipeline(text, voice=self.voice)
            audio_chunks = [audio for _, _, audio in generator]
        
        if not audio_chunks:
            raise ValueError("No audio generated")
        
        full_audio = torch.cat(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
        
        if full_audio.device.type != 'cpu':
            full_audio = full_audio.cpu()
        
        sf.write(output_path, full_audio, self.sample_rate)
        
        duration = len(full_audio) / self.sample_rate
        return duration
    
    def create_person_overlay(self, person_name, duration, position='right'):
        """Create an overlay with person's portrait and name."""
        portrait = self.create_placeholder_portrait(person_name, style='professional')
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
        
        bbox = draw.textbbox((0, 0), person_name, font=font)
        text_width = bbox[2] - bbox[0]
        x = (overlay_width - text_width) // 2
        
        # Glow effect
        for offset in range(3, 0, -1):
            alpha = 80 - offset * 20
            draw.text((x+offset, 342+offset), person_name, fill=(0, 0, 0, alpha), font=font)
        
        # Main text
        draw.text((x, 340), person_name, fill=(255, 255, 255, 255), font=font)
        
        clip = ImageClip(np.array(overlay_img)).set_duration(duration)
        
        if position == 'right':
            clip = clip.set_position((self.resolution[0] - overlay_width - 50, 150))
        else:
            clip = clip.set_position((50, 150))
        
        clip = clip.crossfadein(0.3).crossfadeout(0.3)
        
        return clip
    
    def create_text_overlay(self, segment, duration):
        """Create text overlay with key information."""
        width, height = self.resolution
        overlays = []
        
        # Location tag
        if segment['places']:
            location_text = segment['places'][0]
            location_clip = TextClip(
                location_text,
                fontsize=45,
                color='white',
                font='Arial-Bold',
                bg_color='rgba(200,0,0,0.9)',
                size=(None, None),
                method='caption'
            ).set_position((30, 30)).set_duration(duration)
            overlays.append(location_clip)
        
        # Subtitle
        subtitle_clip = TextClip(
            segment['text'],
            fontsize=36,
            color='white',
            font='Arial',
            bg_color='rgba(0,0,0,0.85)',
            size=(width - 200, None),
            method='caption',
            align='center'
        ).set_position(('center', height - 180)).set_duration(duration)
        
        subtitle_clip = subtitle_clip.crossfadein(0.2)
        overlays.append(subtitle_clip)
        
        return overlays
    
    # def create_news_banner(self, duration, ticker_text="BREAKING NEWS ANALYSIS"):
    #     """Create animated news banner."""
    #     width, height = self.resolution
        
    #     def make_banner(t):
    #         img = Image.new('RGBA', (width, 90), (180, 0, 0, 240))
    #         draw = ImageDraw.Draw(img)
            
    #         try:
    #             font_large = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 38)
    #             font_small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 24)
    #         except:
    #             try:
    #                 font_large = ImageFont.truetype("Arial-Bold", 38)
    #                 font_small = ImageFont.truetype("Arial", 24)
    #             except:
    #                 font_large = font_small = ImageFont.load_default()
            
    #         scroll_offset = int(t * 100) % width
            
    #         draw.text((30, 25), ticker_text, fill='white', font=font_large)
            
    #         ticker = "● LIVE COVERAGE ● LATEST UPDATES ● "
    #         ticker_full = ticker * 10
    #         draw.text((width - scroll_offset, 60), ticker_full, fill='yellow', font=font_small)
            
    #         return np.array(img)
        
    #     banner = VideoClip(make_banner, duration=duration)
    #     return banner.set_position(('center', 0))

    def create_news_banner(self, duration, ticker_text="BREAKING NEWS ANALYSIS"):
        """Create animated news banner."""
        width, height = self.resolution
        
        def make_banner(t):
            # Create RGB image instead of RGBA
            img = Image.new('RGB', (width, 90), (180, 0, 0))
            draw = ImageDraw.Draw(img)
            
            try:
                font_large = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 38)
                font_small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 24)
            except:
                try:
                    font_large = ImageFont.truetype("Arial-Bold", 38)
                    font_small = ImageFont.truetype("Arial", 24)
                except:
                    font_large = font_small = ImageFont.load_default()
            
            scroll_offset = int(t * 100) % width
            
            draw.text((30, 25), ticker_text, fill='white', font=font_large)
            
            ticker = "● LIVE COVERAGE ● LATEST UPDATES ● "
            ticker_full = ticker * 10
            draw.text((width - scroll_offset, 60), ticker_full, fill='yellow', font=font_small)
            
            return np.array(img)
        
        banner = VideoClip(make_banner, duration=duration)
        return banner.set_position(('center', 0))
    
    def create_transition(self, duration=0.5):
        """Create a transition effect between segments."""
        width, height = self.resolution
        
        def make_frame(t):
            progress = t / duration
            img = Image.new('RGB', (width, height), (0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            wipe_pos = int(progress * (width + height))
            
            points = [
                (0, wipe_pos),
                (wipe_pos, 0),
                (wipe_pos + 100, 0),
                (0, wipe_pos + 100)
            ]
            draw.polygon(points, fill=(40, 40, 60))
            
            return np.array(img)
        
        return VideoClip(make_frame, duration=duration)

    def generate_video(self, article_text, output_video_path):
        """Generate complete video from article text with Metal GPU acceleration."""
        print("Parsing article...", file=sys.stderr)
        segments = self.parse_article(article_text)
        
        print(f"Found {len(segments)} segments", file=sys.stderr)
        
        video_clips = []
        audio_clips = []
        current_time = 0
        
        # Enable Metal-optimized video encoding on Mac
        ffmpeg_params = ['-preset', 'medium', '-crf', '18']
        if platform.system() == 'Darwin':
            try:
                import subprocess
                result = subprocess.run(['ffmpeg', '-encoders'], 
                                       capture_output=True, text=True, timeout=2)
                if 'h264_videotoolbox' in result.stdout:
                    print("✓ Using VideoToolbox hardware encoding", file=sys.stderr)
                    ffmpeg_params = [
                        '-c:v', 'h264_videotoolbox',
                        '-b:v', '8000k',
                        '-profile:v', 'high',
                        '-allow_sw', '1'
                    ]
            except:
                pass
        
        for i, segment in enumerate(segments):
            print(f"\n=== Segment {i+1}/{len(segments)} ===", file=sys.stderr)
            
            # Generate audio with GPU acceleration
            audio_path = self.cache_dir / f"segment_{i}.wav"
            duration = self.generate_audio(segment['text'], audio_path)
            
            # Create procedural background
            print(f"Creating procedural background...", file=sys.stderr)
            background = self.get_background_for_segment(segment, duration)
            
            # Darken background slightly for text readability
            background = background.fl_image(lambda img: (img * 0.7).astype('uint8'))
            
            # Create overlays
            text_overlays = self.create_text_overlay(segment, duration)
            banner = self.create_news_banner(duration)
            
            # Add person overlays
            person_overlays = []
            for j, person in enumerate(segment['people'][:2]):
                print(f"Creating portrait for: {person}", file=sys.stderr)
                position = 'right' if j == 0 else 'left'
                person_clip = self.create_person_overlay(person, duration, position)
                person_overlays.append(person_clip)
            
            # Composite video
            all_clips = [background, banner] + person_overlays + text_overlays
            video = CompositeVideoClip(all_clips, size=self.resolution).set_opacity(1)
            video = video.set_duration(duration).set_start(current_time)
            
            # Add transition (except for first segment)
            if i > 0:
                transition = self.create_transition(0.4)
                transition = transition.set_start(current_time - 0.2)
                video_clips.append(transition)
            
            # Load audio
            audio = AudioFileClip(str(audio_path)).set_start(current_time)
            
            video_clips.append(video)
            audio_clips.append(audio)
            current_time += duration
        
        print("\n=== Compositing final video ===", file=sys.stderr)
        
        # Combine all clips
        final_video = CompositeVideoClip(video_clips, size=self.resolution).set_opacity(1)
        
        # Combine audio
        final_audio = concatenate_audioclips([
            AudioFileClip(str(self.cache_dir / f"segment_{i}.wav"))
            for i in range(len(segments))
        ])
        
        final_video = final_video.set_audio(final_audio)
        
        # Write output with hardware acceleration
        print(f"Writing video to {output_video_path}...", file=sys.stderr)
        final_video.write_videofile(
            output_video_path,
            fps=30,
            codec='libx264',
            audio_codec='aac',
            bitrate='8000k',
            temp_audiofile=str(self.cache_dir / 'temp_audio.m4a'),
            remove_temp=True,
            ffmpeg_params=ffmpeg_params,
            threads=8,
            logger=None
        )
        
        print("\n✓ Video generation complete!", file=sys.stderr)

def main():
    if len(sys.argv) < 3:
        print("Usage: ./news_video_gen.py <input_file|-> <output.mp4>", file=sys.stderr)
        print("\n✨ 100% Local & Free - No API Keys Required", file=sys.stderr)
        print("\nFeatures:", file=sys.stderr)
        print("  ✓ Procedural cityscapes and backgrounds", file=sys.stderr)
        print("  ✓ Generated portraits with unique designs", file=sys.stderr)
        print("  ✓ Animated transitions and effects", file=sys.stderr)
        print("  ✓ Metal GPU acceleration (Mac)", file=sys.stderr)
        print("  ✓ VideoToolbox hardware encoding (Mac)", file=sys.stderr)
        print("  ✓ Multi-threaded processing", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Display system info
    print("=== System Configuration ===", file=sys.stderr)
    print(f"Platform: {platform.system()} {platform.machine()}", file=sys.stderr)
    print(f"PyTorch version: {torch.__version__}", file=sys.stderr)
    
    if platform.system() == 'Darwin':
        print(f"Metal available: {torch.backends.mps.is_available()}", file=sys.stderr)
        if torch.backends.mps.is_available():
            print(f"Metal device: {torch.device('mps')}", file=sys.stderr)
    
    print("============================\n", file=sys.stderr)
    
    # Read article text
    if input_file == '-':
        text = sys.stdin.read()
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    
    # Generate video with GPU acceleration
    start_time = time.time()
    generator = NewsVideoGenerator()
    generator.generate_video(text, output_file)
    elapsed = time.time() - start_time
    
    print(f"\n⚡ Total generation time: {elapsed:.1f} seconds", file=sys.stderr)
    print(f"✓ Video saved to: {output_file}", file=sys.stderr)

if __name__ == '__main__':
    main()