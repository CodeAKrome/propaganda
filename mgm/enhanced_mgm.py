#!/usr/bin/env python
"""
Enhanced News Article to Video Generator – 100 % local, free, GPU-accelerated.
Creates narrated news clips with:
  - TTS via Kokoro (Metal/CUDA)
  - Enhanced Stable-Diffusion backgrounds with comprehensive entity and relationship awareness
  - Procedural portraits, banners, transitions
  - Multiple background images per story segment based on entity matches
  - Relationship-aware background generation
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
from collections import defaultdict, Counter

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
# Enhanced Helpers
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
    """Convert entities file into dict with enhanced processing."""
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
            # Clean and expand entity lists
            entities = [x.strip() for x in v.split(",")]
            entities = [e for e in entities if e and len(e) > 1]  # Filter out single chars
            out[k.strip()].extend(entities)
    return out

def parse_story_text(path: str) -> str:
    """Parse story markdown file to extract main content."""
    text = Path(path).read_text(encoding="utf-8")
    # Remove markdown headers and extract main story content
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    # Join all content into single story text
    return ' '.join(lines)

def load_audio_as_array(path: Path, sr: int = 24_000) -> np.ndarray:
    """Load entire audio into RAM, close file-handle immediately."""
    with AudioFileClip(str(path)) as clip:
        arr = clip.to_soundarray(fps=sr)  # (N,2) stereo
        if arr.ndim == 2 and arr.shape[1] == 2:
            arr = arr.mean(axis=1)        # mono
    return arr

# ------------------------------------------------------------------
# Enhanced Core generator
# ------------------------------------------------------------------
class EnhancedNewsVideoGenerator:
    def __init__(self, voice="af_heart", resolution=(1920, 1080)):
        self._setup_metal()
        self.pipeline = KPipeline(lang_code="a")
        self._setup_img_pipe()
        self.voice = voice
        self.resolution = resolution
        self.sample_rate = 24_000
        self.cache_dir = Path("/tmp/video_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Enhanced entity priority weights for background generation
        self.entity_weights = {
            'EVENT': 10,      # Critical - highest priority
            'PERSON': 9,      # High priority
            'ORG': 8,         # High priority
            'GPE': 7,         # Geographic - high priority
            'FAC': 6,         # Facilities - medium priority
            'PRODUCT': 5,     # Products - medium priority
            'NORP': 4,        # Nationalities - medium priority
            'MONEY': 3,       # Money - low priority
            'CARDINAL': 2,    # Numbers - low priority
            'DATE': 1,        # Dates - lowest priority
            'TIME': 1
        }
        
        # NEW: Track entities used for backgrounds to ensure variety
        self.used_background_entities = set()
        
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

    # ---------- Enhanced Entity Detection ----------
    def _find_all_entities_in_sentence(self, sent: str, txt_entities: dict) -> dict:
        """Find all entities mentioned in the sentence with weights."""
        found = {}
        sent_lower = sent.lower()
        
        for ent_type, entity_list in txt_entities.items():
            if ent_type in self.entity_weights:
                found[ent_type] = []
                for entity in entity_list:
                    # Handle multi-word entities
                    words = entity.split()
                    if len(words) == 1:
                        # Single word - use word boundary
                        if re.search(rf'\b{re.escape(entity)}\b', sent, re.I):
                            found[ent_type].append(entity)
                    else:
                        # Multi-word - use phrase search
                        if entity.lower() in sent_lower:
                            found[ent_type].append(entity)
        
        return found

    def _find_relevant_relationships(self, sent: str, relationships: list) -> list:
        """Find relationships where entities are mentioned in the sentence."""
        relevant = []
        sent_lower = sent.lower()
        
        for rel in relationships:
            source, target, relation, description = rel
            
            # Check if source or target is in the sentence
            source_in = any(word in sent_lower for word in source.lower().split())
            target_in = any(word in sent_lower for word in target.lower().split())
            
            if source_in or target_in:
                # Score relationship relevance
                relevance_score = 0
                if source_in and target_in:
                    relevance_score += 10  # Both entities present
                elif source_in or target_in:
                    relevance_score += 5   # One entity present
                
                # Add relation type weight
                relation_lower = relation.lower()
                if 'killed' in relation_lower or 'attacked' in relation_lower:
                    relevance_score += 8  # Violence-related
                elif 'ordered' in relation_lower or 'announced' in relation_lower:
                    relevance_score += 6  # Official actions
                elif 'brokered' in relation_lower or 'mediated' in relation_lower:
                    relevance_score += 4  # Diplomatic
                
                relevant.append((rel, relevance_score))
        
        # Sort by relevance score
        relevant.sort(key=lambda x: x[1], reverse=True)
        return [rel for rel, score in relevant if score > 3]  # Only relevant relationships

    # ---------- Enhanced Background Generation ----------
    def _generate_entity_based_prompts(self, segment: dict, entities: dict, relationships: list) -> list:
        """Generate multiple background prompts based on entities and relationships."""
        prompts = []
        sent_text = segment['text']
        
        # 1. Event-focused backgrounds (highest priority)
        if entities.get('EVENT'):
            for event in entities['EVENT'][:2]:  # Top 2 events
                prompt = f"breaking news background, {event}, dramatic news scene, high detail, 4K"
                prompts.append(prompt)
        
        # 2. People-focused backgrounds
        if entities.get('PERSON'):
            for person in entities['PERSON'][:3]:  # Top 3 people
                prompt = f"professional news background, {person}, official setting, diplomatic atmosphere"
                prompts.append(prompt)
        
        # 3. Location-focused backgrounds
        if entities.get('GPE'):
            for location in entities['GPE'][:2]:  # Top 2 locations
                if 'gaza' in location.lower():
                    prompt = f"middle east conflict background, {location}, war zone, humanitarian crisis"
                elif 'israel' in location.lower():
                    prompt = f"israeli government background, {location}, military briefing room, official atmosphere"
                else:
                    prompt = f"international news background, {location}, diplomatic setting"
                prompts.append(prompt)
        
        # 4. Organization-focused backgrounds
        if entities.get('ORG'):
            for org in entities['ORG'][:2]:  # Top 2 organizations
                if 'hamas' in org.lower():
                    prompt = f"conflict background, {org} headquarters, militant organization, war room"
                elif 'idf' in org.lower() or 'military' in org.lower():
                    prompt = f"military briefing background, {org} command center, military operations"
                elif 'red cross' in org.lower():
                    prompt = f"humanitarian background, {org}, aid workers, medical assistance"
                else:
                    prompt = f"organizational background, {org}, institutional setting"
                prompts.append(prompt)
        
        # 5. Relationship-driven backgrounds
        if relationships:
            for rel in relationships[:2]:  # Top 2 relationships
                source, target, relation, description = rel
                if 'killed' in relation.lower():
                    prompt = f"war casualties background, memorial scene, conflict aftermath, somber atmosphere"
                elif 'attacked' in relation.lower():
                    prompt = f"military action background, conflict zone, battle scene, dramatic news coverage"
                elif 'ordered' in relation.lower():
                    prompt = f"political decision background, government briefing, official announcement, serious atmosphere"
                elif 'brokered' in relation.lower():
                    prompt = f"diplomatic background, peace talks, international mediation, negotiation scene"
                else:
                    prompt = f"news background, {source} and {target}, {relation}, professional setting"
                prompts.append(prompt)
        
        # 6. Context-specific backgrounds based on sentiment and content
        sentiment = segment.get('sentiment', 'neutral')
        if sentiment == 'negative':
            prompts.append(f"crisis news background, conflict zone, emergency situation, dramatic lighting")
        elif sentiment == 'positive':
            prompts.append(f"peace process background, diplomatic talks, hopeful atmosphere, professional setting")
        else:
            prompts.append(f"breaking news background, neutral atmosphere, professional journalism, clean composition")
        
        # 7. Time-sensitive backgrounds
        if entities.get('DATE') or entities.get('TIME'):
            time_context = ""
            if entities.get('TIME'):
                for time_val in entities['TIME']:
                    if 'night' in time_val.lower():
                        time_context = "nighttime news background"
                    elif 'day' in time_val.lower():
                        time_context = "daytime news background"
            
            if time_context:
                prompts.append(time_context)
        
        # Ensure we have at least one prompt
        if not prompts:
            prompts.append(f"news background, {segment.get('topics', ['current events'])[0] if segment.get('topics') else 'current events'}, professional atmosphere")
        
        return prompts[:6]  # Return up to 6 different background prompts



    # ---------- NEW: ultra-specific single prompt generator ----------
    def _build_focused_entity_prompt(self, segment: dict, txt_entities: dict) -> str:
        """
        Create ONE prompt focused on the most important, not-yet-used entity in the sentence.
        """
        if not txt_entities:
            return self._fallback_prompt(segment)

        found = self._find_all_entities_in_sentence(segment["text"], txt_entities)
        
        # Get all entities found in the sentence, sorted by weight
        all_found_entities = []
        for ent_type in sorted(self.entity_weights, key=self.entity_weights.get, reverse=True):
            if ent_type in found:
                for entity_text in found[ent_type]:
                    all_found_entities.append((entity_text, ent_type))

        if not all_found_entities:
            return self._fallback_prompt(segment)

        # Separate into used and unused entities
        unused_entities = [e for e in all_found_entities if e[0] not in self.used_background_entities]
        
        # Prioritize unused entities. If none, fall back to used ones.
        target_entity = None
        if unused_entities:
            # Pick the highest-priority unused entity
            target_entity = unused_entities[0][0]
            self.used_background_entities.add(target_entity)
        else:
            # All entities in this sentence have been used, so just pick the most important one
            target_entity = all_found_entities[0][0]

        # Build a focused prompt around the target entity
        prompt = f"news broadcast background about {target_entity}"
        
        # Add context based on other entities present
        other_entities = [e[0] for e in all_found_entities if e[0] != target_entity]
        if other_entities:
            context_str = ", ".join(other_entities[:2])
            prompt += f", in the context of {context_str}"

        # Add sentiment and quality modifiers
        sentiment = segment.get('sentiment', 'neutral')
        if sentiment == 'negative':
            prompt += ", somber atmosphere, dramatic lighting"
        
        return f"cinematic digital art, {prompt}, highly detailed, 4K, professional news quality"

    def _fallback_prompt(self, segment: dict) -> str:
        """Classic generic prompt when no entities are available."""
        bits = segment.get("topics", []) + segment.get("places", [])
        prompt = ", ".join(bits) if bits else "breaking news"
        return f"cinematic digital art, news background, {prompt}, mood:{segment.get('sentiment','neutral')}, 4K"

    # ---------- OVERRIDE: generate only ONE ultra-specific background ----------
    def get_multiple_backgrounds_for_segment(self, segment: dict, duration: float,
                                           txt_entities: dict = None, relationships: list = None) -> list:
        """
        Returns a list with exactly ONE background clip that is maximally
        specific to the entities appearing in the sentence.
        """
        prompt = self._build_focused_entity_prompt(segment, txt_entities)
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        cached = self.cache_dir / f"bg_{prompt_hash}.png"

        if cached.exists():
            img = Image.open(cached)
        else:
            print(f"🎨 Ultra-specific background: {prompt[:80]}…", file=sys.stderr)
            img = self.image_pipeline(prompt=prompt,
                                      num_inference_steps=2,
                                      guidance_scale=0.0).images[0]
            img.save(cached)

        clip = ImageClip(np.array(img.resize(self.resolution))).set_duration(duration)
        return [clip]          # keep list interface for downstream code




    # def get_multiple_backgrounds_for_segment(self, segment: dict, duration: float,
    #                                        txt_entities: dict = None, relationships: list = None) -> list:
    #     """Generate multiple background images for a single segment."""
    #     if not txt_entities:
    #         # Fallback to simple single background
    #         return [self.create_ai_background(segment, duration)]
        
    #     # Find all entities in this segment
    #     entities = self._find_all_entities_in_sentence(segment['text'], txt_entities)
        
    #     # Find relevant relationships
    #     relevant_relationships = self._find_relevant_relationships(segment['text'], relationships) if relationships else []
        
    #     # Generate multiple prompts
    #     prompts = self._generate_entity_based_prompts(segment, entities, relevant_relationships)
        
    #     background_clips = []
        
    #     for i, prompt in enumerate(prompts):
    #         try:
    #             # Create a sub-duration for each background
    #             sub_duration = duration / len(prompts)
    #             start_time = i * sub_duration
                
    #             # Generate background image
    #             full_prompt = f"cinematic digital art, {prompt}, highly detailed, 4K, professional news quality"
                
    #             prompt_hash = hashlib.md5(full_prompt.encode()).hexdigest()
    #             cached = self.cache_dir / f"bg_{prompt_hash}.png"
                
    #             if cached.exists():
    #                 img = Image.open(cached)
    #             else:
    #                 print(f"🎨 Generating background {i+1}/{len(prompts)}: {prompt[:60]}…", file=sys.stderr)
    #                 img = self.image_pipeline(prompt=full_prompt,
    #                                         num_inference_steps=2,
    #                                         guidance_scale=0.0).images[0]
    #                 img.save(cached)
                
    #             # Create background clip with crossfade
    #             bg_clip = ImageClip(np.array(img.resize(self.resolution))).set_duration(sub_duration)
    #             if len(prompts) > 1:
    #                 # Add crossfade between backgrounds
    #                 if i > 0:
    #                     bg_clip = bg_clip.crossfadein(0.5)
    #                 if i < len(prompts) - 1:
    #                     bg_clip = bg_clip.crossfadeout(0.5)
                
    #             bg_clip = bg_clip.set_start(start_time)
    #             background_clips.append(bg_clip)
                
    #         except Exception as e:
    #             print(f"⚠ Error generating background {i+1}: {e}", file=sys.stderr)
    #             continue
        
    #     # If no backgrounds were generated successfully, create a fallback
    #     if not background_clips:
    #         background_clips = [self.create_ai_background(segment, duration)]
        
    #     return background_clips

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

    # ---------- Enhanced Text parsing ----------
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
        pos, neg = ["approval", "support", "peace", "ceasefire"], ["rejected", "war zones", "killed", "attack", "violation"]
        low = text.lower()
        cpos = sum(low.count(w) for w in pos)
        cneg = sum(low.count(w) for w in neg)
        return "negative" if cneg > cpos else "positive" if cpos > cneg else "neutral"

    # ---- overlays (keeping original implementations) ----
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

    # ---------- Enhanced Final composer ----------
    def generate_enhanced_video(self, article_text: str, output_video_path: str,
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
            print(f"Text: {seg['text'][:100]}...", file=sys.stderr)
            
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

            # Now that we have the exact duration, build the video segment with enhanced backgrounds
            try:
                # Generate multiple backgrounds for this segment
                backgrounds = self.get_multiple_backgrounds_for_segment(seg, dur, txt_entities, relationships)
                
                # Create banner
                banner = self.create_news_banner(dur)
                
                # Create overlays
                overlays = self.create_text_overlay(seg, dur)
                
                # Create person overlays
                person_ov = []
                for j, person in enumerate(seg['people'][:2]):
                    print(f"Creating portrait for: {person}", file=sys.stderr)
                    pos = 'right' if j == 0 else 'left'
                    person_ov.append(self.create_person_overlay(person, dur, pos))

                # Combine all clips
                all_clips = backgrounds + [banner] + person_ov + overlays
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
        print("\n✓ Enhanced video generation complete!", file=sys.stderr)


# ------------------------------------------------------------------
# Enhanced CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Enhanced local news-video generator with multiple backgrounds")
    parser.add_argument("input", help="Article text file, markdown story file, or '-' for stdin")
    parser.add_argument("output", help="Output .mp4")
    parser.add_argument("--data", metavar="ENTITIES.txt",
                        help="Entities metadata file for enhanced background generation")
    parser.add_argument("--report", metavar="REPORT.reporter",
                        help="Report file with relationship data for context-aware backgrounds")
    parser.add_argument("--story", metavar="STORY.md",
                        help="Markdown story file to extract main narrative")
    args = parser.parse_args()

    print("=== Enhanced System ===", file=sys.stderr)
    print(f"Platform: {platform.system()} {platform.machine()}", file=sys.stderr)
    print(f"PyTorch: {torch.__version__}", file=sys.stderr)
    if platform.system() == "Darwin":
        print(f"Metal: {torch.backends.mps.is_available()}", file=sys.stderr)

    # read article/story
    if args.input == "-":
        text = sys.stdin.read()
    elif args.story:
        # Parse story markdown file
        text = parse_story_text(args.story)
    else:
        text = Path(args.input).read_text(encoding="utf-8")

    # parse optional entity and relationship files
    txt_entities = parse_entity_file(args.data) if args.data else None
    relationships = parse_report_file(args.report) if args.report else None
    
    if txt_entities:
        print(f"Loaded entities: {sum(len(v) for v in txt_entities.values())} total", file=sys.stderr)
    if relationships:
        print(f"Loaded relationships: {len(relationships)} total", file=sys.stderr)
    
    # render with enhanced generator
    t0 = time.time()
    gen = EnhancedNewsVideoGenerator()
    gen.generate_enhanced_video(text, args.output, txt_entities, relationships)
    print(f"\n⚡ Total: {time.time() - t0:.1f}s", file=sys.stderr)
    print(f"✓ Saved: {args.output}", file=sys.stderr)
    print(f"🎨 Enhanced with multiple context-aware backgrounds!", file=sys.stderr)


if __name__ == "__main__":
    main()