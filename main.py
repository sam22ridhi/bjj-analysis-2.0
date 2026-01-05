from __future__ import annotations
import os
import time
import shutil
import uuid
import json
import asyncio
import base64
import re
import traceback
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM

# Gemini imports
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# OpenCV
import cv2
import numpy as np

# Configuration
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable required")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable required")

genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="BJJ AI Coach - Hybrid Agentic")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELS ---

class TimestampedEvent(BaseModel):
    time: str
    title: str
    description: str
    category: Optional[str] = "GENERAL"
    frame_image: Optional[str] = None
    frame_timestamp: Optional[str] = None
    model_config = ConfigDict(extra="allow")

class Drill(BaseModel):
    name: str
    focus_area: str
    reason: str
    duration: Optional[str] = "15 min/day"
    frequency: Optional[str] = "5x/week"

class DetailedSkillBreakdown(BaseModel):
    offense: int
    defense: int
    guard: int
    passing: int
    standup: int

class PerformanceGrades(BaseModel):
    defense_grade: str
    offense_grade: str
    control_grade: str

class AnalysisResult(BaseModel):
    overall_score: int
    performance_label: str
    performance_grades: PerformanceGrades
    skill_breakdown: DetailedSkillBreakdown
    strengths: List[str]
    weaknesses: List[str]
    missed_opportunities: List[TimestampedEvent]
    key_moments: List[TimestampedEvent]
    coach_notes: str
    recommended_drills: List[Drill]

db_storage = {}

# --- UTILITIES ---

def parse_time_to_seconds(time_str: str) -> Optional[int]:
    if not time_str:
        return None
    match = re.search(r"(\d{1,2}):(\d{2})", time_str)
    if not match:
        return None
    mm, ss = match.groups()
    return int(mm) * 60 + int(ss)

def find_closest_frame(target_time_sec: int, frames: list) -> dict:
    return min(frames, key=lambda f: abs(f["second"] - target_time_sec))

def attach_frames_to_events(events: List[dict], frames: list):
    for event in events:
        try:
            event_time_sec = parse_time_to_seconds(event.get("time"))
            if event_time_sec is None:
                continue
            closest = find_closest_frame(event_time_sec, frames)
            event["frame_timestamp"] = closest["timestamp"]
            event["frame_image"] = base64.b64encode(closest["bytes"]).decode("utf-8")
        except:
            event["frame_image"] = None

def extract_json_from_text(text: str) -> Dict:
    """Robust JSON extraction"""
    text = text.strip()
    
    # Direct parse
    try:
        return json.loads(text)
    except:
        pass
    
    # Remove markdown
    if "```json" in text or "```" in text:
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            else:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except:
            pass
    
    # Find boundaries
    try:
        start_idx = text.find('{')
        if start_idx == -1:
            raise ValueError("No opening brace")
        
        brace_count = 0
        end_idx = -1
        
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
        
        if end_idx != -1:
            json_str = text[start_idx:end_idx+1]
            return json.loads(json_str)
        
        # Truncation repair
        json_str = text[start_idx:]
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        if open_brackets > close_brackets:
            json_str += ']' * (open_brackets - close_brackets)
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)
        
        return json.loads(json_str)
        
    except:
        pass
    
    raise ValueError("Could not extract JSON")

def is_generic(text: str) -> bool:
    """Check if feedback is too generic"""
    patterns = [r'^More \w+$', r'^Improve \w+$', r'^Work \w+$', r'^Better \w+$']
    for p in patterns:
        if re.match(p, text.strip(), re.IGNORECASE):
            return True
    if not re.search(r'\d{1,2}:\d{2}', text):
        return True
    if len(text) < 20:
        return True
    return False

# --- FRAME EXTRACTION ---

def extract_frames(video_path: str) -> tuple:
    """Extract frames with weighted distribution (40% from end)"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Cannot open video")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        if duration <= 30:
            total_to_extract = 14
        elif duration <= 60:
            total_to_extract = 16
        else:
            total_to_extract = 18
        
        print(f"Extracting {total_to_extract} frames from {duration:.1f}s video")
        
        # Weighted: 25% start, 35% middle, 40% end
        start_frames = max(4, int(total_to_extract * 0.25))
        end_frames = max(6, int(total_to_extract * 0.40))
        middle_frames = total_to_extract - start_frames - end_frames
        
        start_section_end = int(total_frames * 0.20)
        end_section_start = int(total_frames * 0.80)
        
        frames = []
        
        # Extract START
        start_interval = max(1, start_section_end // start_frames)
        for i in range(0, start_section_end, start_interval):
            if len([f for f in frames if f["second"] < duration * 0.20]) >= start_frames:
                break
            frame = get_frame(cap, i, fps)
            if frame:
                frames.append(frame)
        
        # Extract MIDDLE
        middle_section_frames = end_section_start - start_section_end
        middle_interval = max(1, middle_section_frames // middle_frames)
        for i in range(start_section_end, end_section_start, middle_interval):
            if len([f for f in frames if duration * 0.20 <= f["second"] < duration * 0.80]) >= middle_frames:
                break
            frame = get_frame(cap, i, fps)
            if frame:
                frames.append(frame)
        
        # Extract END
        end_section_frames = total_frames - end_section_start
        end_interval = max(1, end_section_frames // end_frames)
        for i in range(end_section_start, total_frames, end_interval):
            if len([f for f in frames if f["second"] >= duration * 0.80]) >= end_frames:
                break
            frame = get_frame(cap, i, fps)
            if frame:
                frames.append(frame)
        
        # Always add last frame
        last = get_frame(cap, total_frames - 1, fps)
        if last and last not in frames:
            frames.append(last)
        
        cap.release()
        frames.sort(key=lambda f: f["second"])
        
        metadata = {
            "duration": round(duration, 2),
            "fps": round(fps, 2),
            "frames_extracted": len(frames),
            "distribution": {"start": start_frames, "middle": middle_frames, "end": end_frames}
        }
        
        print(f"Extracted {len(frames)} frames")
        return frames, metadata
        
    except Exception as e:
        if 'cap' in locals():
            cap.release()
        raise Exception(f"Frame extraction failed: {str(e)}")

def get_frame(cap: cv2.VideoCapture, frame_idx: int, fps: float) -> Optional[dict]:
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            return None
        
        h, w = frame.shape[:2]
        target_h = 720
        target_w = int(w * (target_h / h))
        resized = cv2.resize(frame, (target_w, target_h))
        _, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        timestamp_sec = frame_idx / fps
        timestamp_str = f"{int(timestamp_sec // 60):02d}:{int(timestamp_sec % 60):02d}"
        
        return {
            "bytes": buffer.tobytes(),
            "timestamp": timestamp_str,
            "second": round(timestamp_sec, 2),
            "frame_idx": frame_idx
        }
    except:
        return None

# --- STEP 1: GEMINI VISION EXTRACTION ---

async def extract_frame_observations(frames: List[Dict], user_desc: str, opp_desc: str, duration: float) -> str:
    """Use Gemini to analyze frames and extract observations"""
    
    print("STEP 1: Gemini Vision - Frame Analysis")
    
    try:
        # Build frame list
        frame_list = "\n".join([
            f"Frame {i+1} at {f['timestamp']} ({f['second']}s)"
            for i, f in enumerate(frames)
        ])
        prompt = f"""
You are an expert Brazilian Jiu-Jitsu (BJJ) video analyst performing STRICT FRAME-BY-FRAME PERCEPTION.

YOUR ROLE IS LIMITED TO OBSERVATION.
You do NOT judge performance, assign scores, or give coaching advice.

PRIMARY RULES (NON-NEGOTIABLE):
- You MUST rely ONLY on what is visibly observable in each frame.
- You are FORBIDDEN from assuming intent, pain, referee actions, or outcomes.
- If evidence is unclear or partially visible, you MUST say:
  "Insufficient visual evidence to confirm."

====================
VIDEO CONTEXT
====================
- Duration: {duration}s
- Total Frames: {len(frames)}
- Athlete Being Analyzed (User): {user_desc}
- Opponent: {opp_desc}

====================
REFERENCE KNOWLEDGE (VOCABULARY ONLY)
====================

Use the following terms ONLY if clearly supported by visual evidence.

POSITIONS:
Standing, Clinch,
Closed Guard,
Open Guard (Butterfly, De La Riva, Spider, X-Guard),
Half Guard (Top / Bottom, Knee Shield, Deep Half),
Side Control (Standard, Kesa Gatame, Reverse Kesa),
North-South,
Mount (Low, High, S-Mount),
Back Control (with hooks or body triangle),
Turtle (Top / Bottom).

IMPORTANT POSITION RULE:
- "Full Mount" requires:
  BOTH knees on the mat,
  hips square over opponent’s torso,
  opponent flat on back,
  and NO leg entanglement.
- If ANY condition is missing, DO NOT label as mount.
  Use "Top control (not mount)" or "Transitional position".

ATTACKS & THREATS:
Chokes (RNC, Guillotine, Triangle, Arm Triangle, D'Arce, Anaconda,
Ezekiel, Collar chokes),
Joint Locks (Armbar, Kimura, Americana, Omoplata, Wrist locks),
Leg Locks (Straight Ankle, Kneebar, Heel Hook, Toe Hold, Calf Slicer).

CONTROL INDICATORS (SUPPORTING ONLY, NOT DECISIVE):
- Hip or head control
- Chest-to-chest pressure
- Hooks or body triangle
- Limb isolation
- Flattening opponent
- Opponent forced into defensive posture

====================
SUBMISSION CONFIRMATION (STRICT)
====================

A submission may ONLY be marked if at least ONE is explicitly visible:
- Tapping (hand, foot, or body)
- Match stoppage during a locked submission
- Footage ends immediately during an unmistakably locked submission

Pattern cues alone (leg entanglement, arching, neck control)
are NEVER sufficient.

If unclear → classify as "submission attempt" or "no submission".

====================
FRAME-BY-FRAME TASK
====================

For EACH frame, report exactly:

1. POSITION:
   The clearest dominant or transitional position
   (use conservative labels when unsure).

2. ADVANTAGE:
   User / Opponent / Neutral
   (based ONLY on visible control).

3. ACTION TYPE (SELECT ONE):
   OFFENSE | DEFENSE | GUARD | PASSING | STANDUP | NONE

4. THREATS:
   None / Submission Attempt (name it) / Positional Advance.

5. TECHNICAL DETAILS:
   Observable grips, pressure, transitions, defenses, or escapes.
   Do NOT speculate.

ACTION TYPE DEFINITIONS:
- OFFENSE: Initiated submission attempts or attack chains
- DEFENSE: Escaping, framing, or defending submissions
- GUARD: Bottom-position control, sweeps, or attacks
- PASSING: Clearing legs and advancing past guard
- STANDUP: Takedowns or clinch exchanges
- NONE: Static control or transitions without active skill use

STRICT OUTPUT FORMAT:

Frame X (MM:SS):
[Position] - [Advantage] - [Action Type] - [Threats] - [Technical Details]

====================
CRITICAL FINAL FRAMES (LAST 6–7 ONLY)
====================

Analyze carefully:
- Is a submission CLEARLY locked?
- Is tapping EXPLICITLY visible?
- Does the footage end during control?

DECISION RULE:
- Without tapping or stoppage → NO submission.

====================
FINAL SUMMARY (FACTUAL ONLY)
====================

Provide a short factual summary:

1. OUTCOME VERDICT:
   - Submission: YES / NO
   - Winner: User / Opponent / NONE
   - Technique: <name or NONE>
   - Time: MM:SS or NONE
   - Confidence: HIGH / MEDIUM / LOW
   - Evidence: Brief quote or paraphrase from frames

2. POSITIONAL OVERVIEW:
   - Which positions were clearly established?
   - Who held visible positional control overall?

FINAL CHECK (MANDATORY):
- No submission without explicit evidence
- No "full mount" unless criteria are met
- No techniques not visible in frames
- No coaching, scoring, or evaluation language
"""

        # Prepare content
        content = []
        for f in frames:
            content.append({
                "mime_type": "image/jpeg",
                "data": base64.b64encode(f["bytes"]).decode("utf-8")
            })
        content.append(prompt)
        
        # Call Gemini
        start = time.time()
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 8000
            }
        )
        
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model.generate_content(
                content,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
        )
        
        elapsed = time.time() - start
        print(f"Gemini vision: {elapsed:.2f}s")
        
        # Get text
        try:
            observations = response.text
        except:
            observations = response.candidates[0].content.parts[0].text
        
        return observations
        
    except Exception as e:
        print(f"Vision extraction failed: {e}")
        return f"Error analyzing frames: {str(e)}"

# --- STEP 2: CREWAI AGENTS ---

def create_analysis_crew(observations: str, user_desc: str, opp_desc: str, duration: float):
    """Create CrewAI agents for analysis and formatting"""
    
    # Groq LLM for fast text processing
    llm = LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.2
    )
    
    llm2=LLM(model="gemini/gemini-2.5-flash", temperature=0.2,api_key=GEMINI_API_KEY)
    
    # Agent 1: Technical Analyst
    analyst = Agent(
        role="BJJ Technical Analyst",
        goal=f"Analyze frame observations for {user_desc} to detect submissions, score performance, and identify strengths/weaknesses",
backstory="""
You are a BJJ black belt coach acting as an EVIDENCE-BASED TECHNICAL AUDITOR.

IMPORTANT SCOPE LIMIT:
- You do NOT analyze video or frames directly.
- You ONLY analyze the OBSERVATIONS provided by the vision model.
- The OBSERVATIONS are the single source of truth.

OUTCOME AUTHORITY RULE:
- You MUST accept the Outcome Verdict stated in the OBSERVATIONS.
- You are NOT allowed to override or reinterpret submission decisions.
- If the verdict confidence is MEDIUM or LOW, treat the match as having NO submission.

POSITION AUTHORITY RULE:
- You MUST respect position labels used in OBSERVATIONS.
- You may summarize positional trends but MUST NOT relabel positions.

ALLOWED ACTIONS:
- Aggregate frame-level facts into performance insights
- Score performance based on observed evidence
- Identify strengths, weaknesses, and missed opportunities
- Provide coaching feedback grounded in timestamps

FORBIDDEN ACTIONS:
- Do NOT infer intent, pain, or referee behavior
- Do NOT upgrade control into a submission
- Do NOT introduce techniques not present in OBSERVATIONS
- Do NOT repeat the same issue using different wording

LANGUAGE & SCORING CONSTRAINTS:
- Every claim must reference a timestamp
- Generic phrases are forbidden
- Every score must be justified by at least one timestamp
- If user was submitted: Defense ≤40
- If user finished opponent: Offense ≥80
-NEVER mention frame numbers, frame ranges, or frame indices in the final output.
"""
,
        verbose=True,
        allow_delegation=False,
        llm=llm,
        memory=True
    )
    
    # Agent 2: JSON Formatter
    formatter = Agent(
        role="Data Structure Specialist",
        goal="Convert analysis into valid JSON matching exact schema requirements",
        backstory="""You transform technical analysis into structured JSON. You ensure:
        - Exactly 3 strengths and 3 weaknesses
        - All feedback includes timestamps (MM:SS format)
        - No generic phrases like "More aggression"
        - Scores reflect actual match outcome
        - JSON is valid (no trailing commas, proper syntax)
        """,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    # Task 1: Analysis
    analysis_task = Task(
        description=f"""
Analyze the OBSERVATIONS produced by a frame-by-frame BJJ vision system.

IMPORTANT:
- Do NOT re-detect submissions or re-label positions.
- Your role is to evaluate performance quality based on OBSERVATIONS only.

====================
OBSERVATIONS (AUTHORITATIVE)
====================
{observations}

VIDEO INFO:
- Duration: {duration}s
- User: {user_desc}
- Opponent: {opp_desc}

====================
REQUIRED OUTPUT
====================

1. OUTCOME SUMMARY:
- Restate the outcome exactly as supported by OBSERVATIONS.
- Do NOT modify submission status or technique.

2. SKILL SCORING (0–100, REALISTIC):

Score ONLY what is visibly demonstrated.

⚔️ Offense:
- Initiated submission attempts or attack chains
- NOT positional control

🛡️ Defense:
- Escapes, survival, or defended attacks
- If never threatened → score ≤65

🔒 Guard:
- Effectiveness from bottom positions ONLY
- If guard not meaningfully played → score ≤40

🚶 Passing:
- Clearing legs and advancing past guard
- Holding mount ≠ passing

🧍 Standup:
- Takedowns or clinch exchanges
- If no standing engagement → score = 0

Each score MUST reference at least one timestamp.

3. STRENGTHS (EXACTLY 3):
- Timestamped, technical, non-repetitive
- If submission occurred, Strength #1 MUST be the finish

4. WEAKNESSES (EXACTLY 3):
- Timestamped, distinct technical issues
- If user was submitted, Weakness #1 MUST be the failure

5. MISSED OPPORTUNITIES (2–3):
- Must be visible in OBSERVATIONS
- Positional or submission-chain only

6. COACH NOTES (150–250 words):
- Technical, honest, evidence-based
- No speculation

7. DRILLS (EXACTLY 3):
- Each drill maps directly to a weakness
- Include timestamp justification

FINAL CHECK:
- No contradiction of OBSERVATIONS
- No new techniques
- Scores align with demonstrated actions
"""
,
        agent=analyst,
        expected_output="Detailed technical analysis with submission detection"
    )
    
    # Task 2: JSON Formatting
    formatting_task = Task(
        description="""Convert the analysis into this EXACT JSON structure. NO markdown wrapping.

{{
  "overall_score": <int 0-100>,
  "performance_label": "EXCELLENT|STRONG|SOLID|DEVELOPING|NEEDS IMPROVEMENT",
  "performance_grades": {{
    "defense_grade": "<A+|A|B+|B|C+|C|D+|D>",
    "offense_grade": "<letter>",
    "control_grade": "<letter>"
  }},
  "skill_breakdown": {{
    "offense": <int>,
    "defense": <int>,
    "guard": <int>,
    "passing": <int>,
    "standup": <int>
  }},
  "strengths": [
    "At 0:XX - Specific observation (min 25 chars)",
    "At 0:XX - Another specific observation",
    "At 0:XX - Third specific observation"
  ],
  "weaknesses": [
    "At 0:XX - Specific weakness (min 25 chars)",
    "At 0:XX - Another weakness",
    "At 0:XX - Third weakness"
  ],
  "missed_opportunities": [
    {{"time": "MM:SS", "title": "Brief", "description": "Detail", "category": "SUBMISSION|POSITION|SWEEP"}}
  ],
  "key_moments": [
    {{"time": "MM:SS", "title": "Event", "description": "What happened", "category": "SUBMISSION|TRANSITION|DEFENSE"}}
  ],
  "coach_notes": "Paragraph 150-250 words",
  "recommended_drills": [
    {{"name": "Drill 1", "focus_area": "Area", "reason": "Why at timestamp", "duration": "15 min/day", "frequency": "5x/week"}},
    {{"name": "Drill 2", "focus_area": "Area", "reason": "Why", "duration": "10 min/day", "frequency": "4x/week"}},
    {{"name": "Drill 3", "focus_area": "Area", "reason": "Why", "duration": "12 min/day", "frequency": "3x/week"}}
  ]
}}

VALIDATION:
- All timestamps in MM:SS format
- No trailing commas
- Exactly 3 strengths, 3 weaknesses, 3 drills
- All feedback includes timestamps
- No generic phrases
""",
        agent=formatter,
        expected_output="Valid JSON only"
    )
    
    # Create crew
    crew = Crew(
        agents=[analyst, formatter],
        tasks=[analysis_task, formatting_task],
        process=Process.sequential,
        verbose=True
    )
    
    return crew

# --- HYBRID ANALYSIS ---

async def hybrid_agentic_analysis(
    frames: List[Dict],
    metadata: Dict,
    user_desc: str,
    opp_desc: str,
    activity_type: str,
    analysis_id: str = None
) -> AnalysisResult:
    """Hybrid: Gemini vision + CrewAI agents + Python validation"""
    
    print("\n" + "="*70)
    print("HYBRID AGENTIC ANALYSIS")
    print("="*70)
    
    try:
        if analysis_id:
            db_storage[analysis_id]["progress"] = 30
        
        # STEP 1: Gemini Vision
        observations = await extract_frame_observations(
            frames, user_desc, opp_desc, metadata["duration"]
        )
        
        if analysis_id:
            db_storage[analysis_id]["progress"] = 60
        
        # STEP 2: CrewAI Agents
        print("\nSTEP 2: CrewAI Agents - Analysis & Formatting")
        crew = create_analysis_crew(observations, user_desc, opp_desc, metadata["duration"])
        
        crew_start = time.time()
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            crew.kickoff
        )
        crew_time = time.time() - crew_start
        print(f"CrewAI: {crew_time:.2f}s")
        
        if analysis_id:
            db_storage[analysis_id]["progress"] = 90
        
        # STEP 3: Parse & Validate
        print("\nSTEP 3: Python Validation")
        result_text = str(result)
        
        # Clean markdown
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        data = extract_json_from_text(result_text)
        data = validate_and_filter(data, frames)
        
        # Attach frames
        attach_frames_to_events(data.get("missed_opportunities", []), frames)
        attach_frames_to_events(data.get("key_moments", []), frames)
        
        if analysis_id:
            db_storage[analysis_id]["progress"] = 100
        
        print("Analysis complete")
        print("="*70 + "\n")
        
        return AnalysisResult(**data)
        
    except Exception as e:
        print(f"Hybrid analysis failed: {e}")
        traceback.print_exc()
        fallback = make_fallback(frames)
        if analysis_id:
            db_storage[analysis_id]["used_fallback"] = True
        return AnalysisResult(**fallback)

def validate_and_filter(data: Dict, frames: List[Dict]) -> Dict:
    """Python-level validation and generic filtering"""
    
    # Validate scores
    if "overall_score" not in data:
        data["overall_score"] = 65
    data["overall_score"] = max(0, min(100, data["overall_score"]))
    
    if "performance_label" not in data:
        score = data["overall_score"]
        if score >= 85:
            data["performance_label"] = "EXCELLENT PERFORMANCE"
        elif score >= 75:
            data["performance_label"] = "STRONG PERFORMANCE"
        elif score >= 60:
            data["performance_label"] = "SOLID PERFORMANCE"
        else:
            data["performance_label"] = "DEVELOPING PERFORMANCE"
    
    if "performance_grades" not in data:
        data["performance_grades"] = {"defense_grade": "C+", "offense_grade": "C", "control_grade": "C+"}
    
    if "skill_breakdown" not in data:
        base = data["overall_score"]
        data["skill_breakdown"] = {
            "offense": max(0, min(100, base - 5)),
            "defense": max(0, min(100, base + 3)),
            "guard": max(0, min(100, base - 2)),
            "passing": max(0, min(100, base - 10)),
            "standup": max(0, min(100, base - 13))
        }
    
    # Filter generic feedback
    for field in ["strengths", "weaknesses"]:
        if field in data and data[field]:
            filtered = [item for item in data[field] if not is_generic(item)]
            if len(filtered) >= 3:
                data[field] = filtered[:3]
            else:
                data[field] = make_specific(field, frames, filtered)
        else:
            data[field] = make_specific(field, frames, [])
    
    # Validate other fields
    if "missed_opportunities" not in data or not data["missed_opportunities"]:
        data["missed_opportunities"] = [{
            "time": frames[len(frames)//2]["timestamp"],
            "title": "Position",
            "description": "Review",
            "category": "POSITION"
        }]
    
    if "key_moments" not in data or not data["key_moments"]:
        data["key_moments"] = [{
            "time": frames[-3]["timestamp"],
            "title": "Exchange",
            "description": "Work",
            "category": "TRANSITION"
        }]
    
    if "coach_notes" not in data or len(data["coach_notes"]) < 50:
        data["coach_notes"] = "Focus on techniques. Review timestamps for improvement."
    
    if "recommended_drills" not in data or len(data["recommended_drills"]) < 3:
        data["recommended_drills"] = [
            {"name": "Control", "focus_area": "General", "reason": "Improve awareness", "duration": "15 min/day", "frequency": "5x/week"},
            {"name": "Guard", "focus_area": "Defense", "reason": "Strengthen defense", "duration": "10 min/day", "frequency": "4x/week"},
            {"name": "Flow", "focus_area": "Movement", "reason": "Improve transitions", "duration": "12 min/day", "frequency": "3x/week"}
        ]
    
    return data

def make_specific(field: str, frames: List[Dict], existing: List[str]) -> List[str]:
    feedback = existing.copy()
    
    start = frames[len(frames) // 6]
    mid = frames[len(frames) // 2]
    end = frames[-2] if len(frames) > 1 else frames[-1]
    
    if field == "strengths":
        templates = [
            f"At {start['timestamp']} - Maintained structure in opening",
            f"At {mid['timestamp']} - Showed awareness during exchange",
            f"At {end['timestamp']} - Demonstrated control"
        ]
    else:
        templates = [
            f"At {start['timestamp']} - Could improve positioning",
            f"At {mid['timestamp']} - Slow to recognize opportunity",
            f"At {end['timestamp']} - Room to improve execution"
        ]
    
    for t in templates:
        if len(feedback) < 3:
            feedback.append(t)
    
    return feedback[:3]

def make_fallback(frames: List[Dict]) -> Dict:
    mid = frames[len(frames)//2]["timestamp"]
    end = frames[-2]["timestamp"] if len(frames) > 1 else frames[-1]["timestamp"]
    
    return {
        "overall_score": 65,
        "performance_label": "SOLID PERFORMANCE",
        "performance_grades": {"defense_grade": "C+", "offense_grade": "C", "control_grade": "C+"},
        "skill_breakdown": {"offense": 60, "defense": 68, "guard": 63, "passing": 55, "standup": 52},
        "strengths": [
            f"At 0:10 - Maintained structure",
            f"At {mid} - Showed awareness",
            f"At {end} - Demonstrated control"
        ],
        "weaknesses": [
            f"At 0:15 - Could improve positioning",
            f"At {mid} - Slow to recognize opportunity",
            f"At {end} - Room to improve execution"
        ],
        "missed_opportunities": [{"time": mid, "title": "Position", "description": "Review", "category": "POSITION"}],
        "key_moments": [{"time": end, "title": "Exchange", "description": "Work", "category": "TRANSITION"}],
        "coach_notes": "Focus on techniques. Review timestamps.",
        "recommended_drills": [
            {"name": "Control", "focus_area": "General", "reason": "Improve", "duration": "15 min/day", "frequency": "5x/week"},
            {"name": "Guard", "focus_area": "Defense", "reason": "Strengthen", "duration": "10 min/day", "frequency": "4x/week"},
            {"name": "Flow", "focus_area": "Movement", "reason": "Improve", "duration": "12 min/day", "frequency": "3x/week"}
        ]
    }

# --- API ---

@app.post("/analyze-complete")
async def analyze_complete(
    file: UploadFile = File(...),
    user_description: str = Form(...),
    opponent_description: str = Form(...),
    activity_type: str = Form("Brazilian Jiu-Jitsu")
):
    start_time = time.time()
    file_path = None
    
    try:
        file_name = f"{uuid.uuid4()}_{file.filename}"
        file_path = f"temp_videos/{file_name}"
        os.makedirs("temp_videos", exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        analysis_id = str(uuid.uuid4())
        db_storage[analysis_id] = {"status": "processing", "progress": 0}
        
        # Extract frames
        frames, metadata = await asyncio.get_event_loop().run_in_executor(
            None, extract_frames, file_path
        )
        
        # Hybrid analysis
        result = await hybrid_agentic_analysis(
            frames, metadata,
            user_description.strip(), opponent_description.strip(),
            activity_type, analysis_id
        )
        
        total_time = time.time() - start_time
        
        return {
            "status": "completed",
            "data": result.model_dump(),
            "processing_time": f"{total_time:.2f}s",
            "used_fallback": db_storage[analysis_id].get("used_fallback", False),
            "method": "hybrid_agentic"
        }
    except Exception as e:
        print(f"Error: {e}")
        try:
            frames_fb, _ = await asyncio.get_event_loop().run_in_executor(None, extract_frames, file_path)
            fallback = make_fallback(frames_fb)
        except:
            fallback = make_fallback([{"timestamp": "00:30", "second": 30}])
        
        return {
            "status": "completed_with_fallback",
            "data": fallback,
            "error": str(e),
            "used_fallback": True
        }
    finally:
        if file_path:
            try:
                os.remove(file_path)
            except:
                pass

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "26.0.0-hybrid-agentic"}

@app.get("/")
async def root():
    return {
        "message": "BJJ AI Coach - Hybrid Agentic",
        "version": "26.0.0",
        "architecture": "Gemini Vision + CrewAI Agents + Python Validation",
        "agents": {
            "gemini": "Frame-by-frame vision analysis",
            "analyst_agent": "Technical analysis + submission detection (Groq)",
            "formatter_agent": "JSON structure + validation (Groq)",
            "python": "Generic filtering + frame attachment"
        },
        "benefits": [
            "Gemini's vision for accurate frame analysis",
            "Groq's speed for text processing (3-5x faster)",
            "Multi-agent review for quality",
            "Python guardrails against generic feedback"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
