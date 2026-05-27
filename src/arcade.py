"""Arcade-style GUI for the multi-agent research system.

A Pygame scene with the 6 agents as pixel characters arranged on a tiled floor in
the same topology as the LangGraph wiring. The LangGraph pipeline runs in a
background thread and emits events into a thread-safe queue; the Pygame main
loop drains the queue and animates agent states + flowing arrows accordingly.

The existing graph code (src/graph.py, src/agents/*, etc.) is NOT modified —
this file constructs a parallel graph instance with the same node functions
wrapped in an event-emitting shim.

Run with:
    python -m src.arcade
"""

import logging
import queue
import sys
import threading
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ensure project root is on sys.path so `python -m src.arcade` works
sys.path.insert(0, str(Path(__file__).parent.parent))

import pygame
from langgraph.graph import StateGraph, START, END

import src  # noqa: F401 — triggers truststore.inject_into_ssl()
from src.state import AgentState
from src.agents.researcher import researcher_pro_node, researcher_skeptical_node
from src.agents.critic import critic_node
from src.agents.synthesizer import synthesizer_node
from src.agents.fact_checker import fact_checker_node
from src.agents.devils_advocate import devils_advocate_node
from src.conflict.resolution import critic_router, post_synthesis_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("arcade")


# ─── PIPELINE WRAPPING ────────────────────────────────────────────────────────

EVENT_Q: queue.Queue = queue.Queue()


def _wrap(name: str, fn):
    """Wrap a node fn so it emits start/done events into the shared queue."""
    def wrapped(state):
        EVENT_Q.put(("start", name, None))
        try:
            result = fn(state)
        except Exception as exc:
            EVENT_Q.put(("error", name, str(exc)))
            raise
        EVENT_Q.put(("done", name, result))
        return result
    return wrapped


def build_arcade_graph():
    """Same wiring as src/graph.py, but with event-emitting wrappers."""
    wf = StateGraph(AgentState)
    wf.add_node("researcher_pro", _wrap("researcher_pro", researcher_pro_node))
    wf.add_node("researcher_skeptical", _wrap("researcher_skeptical", researcher_skeptical_node))
    wf.add_node("critic", _wrap("critic", critic_node))
    wf.add_node("synthesizer", _wrap("synthesizer", synthesizer_node))
    wf.add_node("fact_checker", _wrap("fact_checker", fact_checker_node))
    wf.add_node("devils_advocate", _wrap("devils_advocate", devils_advocate_node))

    wf.add_edge(START, "researcher_pro")
    wf.add_edge(START, "researcher_skeptical")
    wf.add_edge("researcher_pro", "critic")
    wf.add_edge("researcher_skeptical", "critic")
    wf.add_conditional_edges("critic", critic_router, {
        "researcher_pro": "researcher_pro",
        "researcher_skeptical": "researcher_skeptical",
        "synthesizer": "synthesizer",
    })
    wf.add_edge("synthesizer", "fact_checker")
    wf.add_edge("fact_checker", "devils_advocate")
    wf.add_conditional_edges("devils_advocate", post_synthesis_router, {
        "synthesizer": "synthesizer",
        "end": END,
    })
    return wf.compile()


def run_pipeline(query: str):
    """Background-thread entry point. Runs the graph and emits a 'complete' event."""
    graph = build_arcade_graph()
    initial = {
        "query": query,
        "research_findings_pro": "",
        "research_findings_skeptical": "",
        "research_sources_pro": [],
        "research_sources_skeptical": [],
        "critique": {},
        "synthesis": "",
        "fact_check": {},
        "devils_advocate": {},
        "research_iteration": 0,
        "synthesis_iteration": 0,
        "final_answer": "",
    }
    try:
        result = graph.invoke(initial)
        EVENT_Q.put(("complete", None, result))
    except Exception as exc:
        logger.exception("Pipeline crashed")
        EVENT_Q.put(("crashed", None, str(exc)))


# ─── VISUAL CONSTANTS ─────────────────────────────────────────────────────────

WIDTH, HEIGHT = 1400, 900
TILE_SIZE = 40
FPS = 30

# Palette — retro arcade
BG_DARK = (24, 24, 36)
BG_TILE_A = (32, 32, 48)
BG_TILE_B = (40, 40, 60)
GRID_LINE = (52, 52, 80)
TEXT_LIGHT = (235, 235, 245)
TEXT_DIM = (155, 155, 175)
ACCENT_GOLD = (255, 215, 0)
PANEL_BG = (20, 20, 30)
PANEL_BORDER = (90, 90, 130)

# Per-agent style
@dataclass
class AgentStyle:
    pos: tuple[int, int]       # center
    color: tuple[int, int, int]
    accent: tuple[int, int, int]
    display: str
    icon: str                  # which accessory drawing
    role: str

AGENTS: dict[str, AgentStyle] = {
    "researcher_pro": AgentStyle(
        pos=(280, 260), color=(34, 197, 94), accent=(187, 247, 208),
        display="PRO", icon="magnifier", role="Researcher · supporting evidence",
    ),
    "researcher_skeptical": AgentStyle(
        pos=(1120, 260), color=(239, 68, 68), accent=(254, 202, 202),
        display="SKEPTIC", icon="scroll", role="Researcher · counter-evidence",
    ),
    "critic": AgentStyle(
        pos=(700, 400), color=(245, 158, 11), accent=(254, 240, 138),
        display="CRITIC", icon="gavel", role="Rubric judge",
    ),
    "synthesizer": AgentStyle(
        pos=(700, 560), color=(139, 92, 246), accent=(221, 214, 254),
        display="SYNTH", icon="quill", role="Mediator",
    ),
    "fact_checker": AgentStyle(
        pos=(500, 720), color=(6, 182, 212), accent=(165, 243, 252),
        display="FACT-CHECK", icon="monocle", role="Chain-of-Verification",
    ),
    "devils_advocate": AgentStyle(
        pos=(900, 720), color=(220, 38, 127), accent=(251, 207, 232),
        display="DEVIL", icon="horns", role="Red-team",
    ),
}

CONNECTIONS = [
    # (src, dst, kind)  kind: "forward" (always solid) or "loop" (revise — dashed when active)
    ("researcher_pro", "critic", "forward"),
    ("researcher_skeptical", "critic", "forward"),
    ("critic", "synthesizer", "forward"),
    ("synthesizer", "fact_checker", "forward"),
    ("fact_checker", "devils_advocate", "forward"),
    ("critic", "researcher_pro", "loop"),
    ("critic", "researcher_skeptical", "loop"),
    ("devils_advocate", "synthesizer", "loop"),
]


# ─── PIXEL CHARACTER DRAWING ──────────────────────────────────────────────────

PIX = 4  # size of one pixel in the character (so a 16x16 character is 64x64 px)


def _px(surf, x, y, color):
    pygame.draw.rect(surf, color, (x, y, PIX, PIX))


def draw_character(surf, cx, cy, style: AgentStyle, state: str, anim_t: float):
    """Draw a 16x20 pixel character centered at (cx, cy)."""
    # Bounce when working
    bounce = int(3 * abs(pygame.math.Vector2(0, 0).rotate(0).x)) if False else 0
    if state == "working":
        bounce = int(4 * abs(((anim_t * 6) % 2) - 1))
    cy -= bounce

    # Character grid: 16 wide × 20 tall pixels of size PIX
    w, h = 16 * PIX, 20 * PIX
    ox = cx - w // 2
    oy = cy - h // 2

    color = style.color
    accent = style.accent
    skin = (255, 220, 178)
    dark = (40, 30, 30)

    # Drop shadow (oval beneath)
    shadow_surf = pygame.Surface((w + 12, 10), pygame.SRCALPHA)
    pygame.draw.ellipse(shadow_surf, (0, 0, 0, 90), shadow_surf.get_rect())
    surf.blit(shadow_surf, (ox - 6, oy + h - 4))

    # Body (rows 8-16 — torso/legs)
    body_top = oy + 8 * PIX
    body_bot = oy + 16 * PIX
    pygame.draw.rect(surf, color, (ox + 3 * PIX, body_top, 10 * PIX, body_bot - body_top))

    # Accent band on body
    pygame.draw.rect(surf, accent, (ox + 3 * PIX, body_top + 3 * PIX, 10 * PIX, PIX))

    # Legs (rows 16-20)
    pygame.draw.rect(surf, dark, (ox + 4 * PIX, oy + 16 * PIX, 3 * PIX, 4 * PIX))
    pygame.draw.rect(surf, dark, (ox + 9 * PIX, oy + 16 * PIX, 3 * PIX, 4 * PIX))

    # Head (rows 0-8)
    head_x = ox + 4 * PIX
    head_y = oy
    head_w = 8 * PIX
    head_h = 8 * PIX
    pygame.draw.rect(surf, skin, (head_x, head_y, head_w, head_h))

    # Eyes (open if active/working, closed if idle)
    eye_y = head_y + 3 * PIX
    if state == "idle":
        # closed eyes — small dashes
        pygame.draw.rect(surf, dark, (head_x + 2 * PIX, eye_y + PIX, PIX, PIX // 2 or 1))
        pygame.draw.rect(surf, dark, (head_x + 5 * PIX, eye_y + PIX, PIX, PIX // 2 or 1))
    else:
        pygame.draw.rect(surf, dark, (head_x + 2 * PIX, eye_y, PIX, PIX))
        pygame.draw.rect(surf, dark, (head_x + 5 * PIX, eye_y, PIX, PIX))
        # blinking pupil glow on active
        if state == "working":
            glow = (255, 255, 200) if int(anim_t * 4) % 2 == 0 else dark
            pygame.draw.rect(surf, glow, (head_x + 2 * PIX, eye_y, PIX, PIX))
            pygame.draw.rect(surf, glow, (head_x + 5 * PIX, eye_y, PIX, PIX))

    # Mouth
    if state == "done":
        pygame.draw.rect(surf, dark, (head_x + 3 * PIX, head_y + 5 * PIX, 2 * PIX, PIX))
    else:
        pygame.draw.rect(surf, dark, (head_x + 3 * PIX, head_y + 5 * PIX, 2 * PIX, PIX // 2 or 1))

    # Accessory / icon (drawn over/around the character based on role)
    _draw_accessory(surf, ox, oy, style.icon, color, accent, anim_t, state)

    # Status ring (under feet)
    ring_color = {
        "idle": (60, 60, 90),
        "working": ACCENT_GOLD,
        "done": (100, 220, 120),
        "error": (240, 80, 80),
    }.get(state, (60, 60, 90))
    pygame.draw.circle(surf, ring_color, (cx, oy + h + 6), 18, 2)
    if state == "working":
        # pulsing inner ring
        r = 14 + int(3 * abs(((anim_t * 3) % 2) - 1))
        pygame.draw.circle(surf, ring_color, (cx, oy + h + 6), r, 1)


def _draw_accessory(surf, ox, oy, icon, color, accent, t, state):
    """Draw the role-specific icon on/near the character."""
    if icon == "magnifier":
        # Held magnifying glass to the right of body
        gx, gy = ox + 14 * PIX, oy + 11 * PIX
        pygame.draw.circle(surf, accent, (gx, gy), 8, 2)
        pygame.draw.line(surf, accent, (gx + 5, gy + 5), (gx + 12, gy + 12), 2)
    elif icon == "scroll":
        # Scroll in hand
        gx, gy = ox + 14 * PIX, oy + 10 * PIX
        pygame.draw.rect(surf, (245, 232, 180), (gx, gy, 10, 14))
        pygame.draw.rect(surf, (140, 100, 50), (gx, gy, 10, 14), 1)
        for i in range(3):
            pygame.draw.line(surf, (140, 100, 50), (gx + 2, gy + 3 + i * 3), (gx + 8, gy + 3 + i * 3))
    elif icon == "gavel":
        # Gavel above head — bobs when working
        gx = ox + 8 * PIX
        gy_offset = -int(4 * abs(((t * 4) % 2) - 1)) if state == "working" else 0
        gy = oy - 8 + gy_offset
        pygame.draw.rect(surf, (200, 150, 100), (gx - 8, gy, 16, 8))
        pygame.draw.rect(surf, (140, 90, 50), (gx - 8, gy, 16, 8), 1)
        pygame.draw.rect(surf, (140, 90, 50), (gx - 1, gy + 8, 2, 10))
    elif icon == "quill":
        # Quill pen extending up-right
        gx, gy = ox + 14 * PIX, oy + 10 * PIX
        pygame.draw.line(surf, accent, (gx, gy + 10), (gx + 8, gy - 6), 3)
        pygame.draw.line(surf, (220, 220, 240), (gx + 8, gy - 6), (gx + 12, gy - 12), 2)
    elif icon == "monocle":
        # Monocle over right eye
        head_x = ox + 4 * PIX
        head_y = oy
        eye_y = head_y + 3 * PIX
        pygame.draw.circle(surf, accent, (head_x + 5 * PIX + PIX // 2, eye_y + PIX // 2), 6, 2)
        pygame.draw.line(surf, accent, (head_x + 7 * PIX, eye_y + 2 * PIX), (head_x + 8 * PIX, eye_y + 4 * PIX), 1)
    elif icon == "horns":
        # Devil horns on top of head
        head_x = ox + 4 * PIX
        head_y = oy
        pygame.draw.polygon(surf, color, [
            (head_x + PIX, head_y),
            (head_x + 3 * PIX, head_y - 8),
            (head_x + 4 * PIX, head_y),
        ])
        pygame.draw.polygon(surf, color, [
            (head_x + 4 * PIX, head_y),
            (head_x + 5 * PIX, head_y - 8),
            (head_x + 7 * PIX, head_y),
        ])
        # Tail
        tail_x = ox + 3 * PIX
        tail_y = oy + 16 * PIX
        pygame.draw.line(surf, color, (tail_x, tail_y), (tail_x - 6, tail_y + 4), 3)
        pygame.draw.line(surf, color, (tail_x - 6, tail_y + 4), (tail_x - 4, tail_y + 10), 3)


# ─── BACKGROUND + ARROWS ──────────────────────────────────────────────────────

def draw_tiled_floor(surf):
    """Checkered grid background with subtle scanlines."""
    surf.fill(BG_DARK)
    for y in range(0, HEIGHT, TILE_SIZE):
        for x in range(0, WIDTH, TILE_SIZE):
            color = BG_TILE_A if ((x // TILE_SIZE + y // TILE_SIZE) % 2 == 0) else BG_TILE_B
            pygame.draw.rect(surf, color, (x, y, TILE_SIZE, TILE_SIZE))
    # Grid lines
    for y in range(0, HEIGHT, TILE_SIZE):
        pygame.draw.line(surf, GRID_LINE, (0, y), (WIDTH, y), 1)
    for x in range(0, WIDTH, TILE_SIZE):
        pygame.draw.line(surf, GRID_LINE, (x, 0), (x, HEIGHT), 1)


def draw_arrow(surf, start, end, color, dashed=False, animated=False, t=0.0):
    """Draw an arrow from start to end, optionally dashed and animated."""
    sx, sy = start
    ex, ey = end
    vec = pygame.math.Vector2(ex - sx, ey - sy)
    length = vec.length()
    if length < 1:
        return
    direction = vec.normalize()
    # Pull endpoints inward so they don't overlap the character sprites
    inset = 50
    sx += int(direction.x * inset)
    sy += int(direction.y * inset)
    ex -= int(direction.x * inset)
    ey -= int(direction.y * inset)

    if dashed:
        dash_len = 12
        gap = 6
        total = dash_len + gap
        offset = (t * 50) % total if animated else 0
        dist = 0
        while dist < length - 2 * inset:
            d1 = max(0, dist - offset)
            d2 = min(length - 2 * inset, dist + dash_len - offset)
            if d2 > d1:
                p1 = (sx + direction.x * d1, sy + direction.y * d1)
                p2 = (sx + direction.x * d2, sy + direction.y * d2)
                pygame.draw.line(surf, color, p1, p2, 3)
            dist += total
    else:
        pygame.draw.line(surf, color, (sx, sy), (ex, ey), 3)
        if animated:
            # Flowing dot along the line
            dot_t = (t * 0.4) % 1.0
            dx = sx + direction.x * dot_t * (length - 2 * inset)
            dy = sy + direction.y * dot_t * (length - 2 * inset)
            pygame.draw.circle(surf, ACCENT_GOLD, (int(dx), int(dy)), 6)
            pygame.draw.circle(surf, color, (int(dx), int(dy)), 6, 1)

    # Arrowhead
    head_len = 12
    head_w = 8
    perp = pygame.math.Vector2(-direction.y, direction.x)
    p_tip = (ex, ey)
    p_left = (ex - direction.x * head_len + perp.x * head_w,
              ey - direction.y * head_len + perp.y * head_w)
    p_right = (ex - direction.x * head_len - perp.x * head_w,
               ey - direction.y * head_len - perp.y * head_w)
    pygame.draw.polygon(surf, color, [p_tip, p_left, p_right])


# ─── SPEECH BUBBLE ────────────────────────────────────────────────────────────

def draw_speech_bubble(surf, font_small, font_tiny, anchor, text, max_width=320):
    """Draw a pixel-style speech bubble above/beside an agent."""
    ax, ay = anchor
    lines = []
    for paragraph in text.split("\n"):
        wrapped = textwrap.wrap(paragraph, width=42) or [""]
        lines.extend(wrapped)
    lines = lines[:6]  # cap

    line_height = font_small.get_linesize()
    text_w = max(font_small.size(line)[0] for line in lines) if lines else 100
    text_h = line_height * len(lines)
    pad = 10
    bw = min(max_width, text_w + 2 * pad)
    bh = text_h + 2 * pad

    # Position bubble above-right of anchor by default
    bx = ax + 30
    by = ay - bh - 30
    if bx + bw > WIDTH - 20:
        bx = ax - bw - 30
    if by < 80:
        by = ay + 30

    # Outer shadow
    shadow = pygame.Surface((bw + 8, bh + 8), pygame.SRCALPHA)
    pygame.draw.rect(shadow, (0, 0, 0, 120), (4, 4, bw, bh), border_radius=6)
    surf.blit(shadow, (bx - 4, by - 4))

    # Body
    pygame.draw.rect(surf, (250, 250, 245), (bx, by, bw, bh), border_radius=6)
    pygame.draw.rect(surf, (40, 40, 60), (bx, by, bw, bh), 2, border_radius=6)

    # Tail toward anchor
    tail_anchor_x = max(bx + 12, min(ax, bx + bw - 12))
    if by + bh < ay:
        tail = [(tail_anchor_x - 8, by + bh), (tail_anchor_x + 8, by + bh), (ax, ay - 30)]
    else:
        tail = [(tail_anchor_x - 8, by), (tail_anchor_x + 8, by), (ax, ay + 30)]
    pygame.draw.polygon(surf, (250, 250, 245), tail)
    pygame.draw.lines(surf, (40, 40, 60), False, [tail[0], tail[2], tail[1]], 2)

    # Text
    for i, line in enumerate(lines):
        surf.blit(font_small.render(line, True, (30, 30, 50)), (bx + pad, by + pad + i * line_height))


# ─── AGENT STATE TRACKING ─────────────────────────────────────────────────────

@dataclass
class AgentRuntime:
    name: str
    style: AgentStyle
    state: str = "idle"  # idle, working, done, error
    last_event_t: float = 0.0
    bubble_text: str = ""
    bubble_until: float = 0.0  # show bubble until this time
    fire_count: int = 0


SOURCE_BADGES = {
    "document_retriever": "[RAG]",
    "tavily_search_results_json": "[WEB·T]",
    "duckduckgo_search": "[WEB·DDG]",
    "python_repl": "[PY]",
}


def _format_sources(sources: list[str]) -> str:
    """Convert a list of tool names into deduplicated short badges."""
    unique = list(dict.fromkeys(sources))
    if not unique:
        return "no tools"
    return " ".join(SOURCE_BADGES.get(s, s) for s in unique)


def summarize_output(name: str, updates: dict) -> str:
    """Produce a short, role-specific speech-bubble summary from node updates."""
    if name == "researcher_pro":
        findings = updates.get("research_findings_pro", "")
        sources = updates.get("research_sources_pro", [])
        return f"Sources: {_format_sources(sources)}\n\"{(findings[:80] or '').strip()}…\""
    if name == "researcher_skeptical":
        findings = updates.get("research_findings_skeptical", "")
        sources = updates.get("research_sources_skeptical", [])
        return f"Sources: {_format_sources(sources)}\n\"{(findings[:80] or '').strip()}…\""
    if name == "critic":
        crit = updates.get("critique", {})
        verdict = crit.get("verdict", "?").upper()
        rubric = crit.get("rubric", {})
        rub_str = " ".join(f"{k[:3]}:{v}" for k, v in rubric.items())
        return f"Verdict: {verdict}\n{rub_str}"
    if name == "synthesizer":
        synth = updates.get("synthesis", "")
        return f"Drafted answer ({len(synth)} chars)\n\"{(synth[:80] or '').strip()}…\""
    if name == "fact_checker":
        fc = updates.get("fact_check", {})
        verdict = fc.get("verdict", "?").upper()
        n_claims = len(fc.get("claims", []))
        n_issues = len(fc.get("issues", []))
        return f"Verdict: {verdict}\nChecked {n_claims} claims, {n_issues} issues."
    if name == "devils_advocate":
        da = updates.get("devils_advocate", {})
        verdict = da.get("verdict", "?").upper()
        n = len(da.get("weaknesses", []))
        return f"Verdict: {verdict}\nFound {n} weakness{'es' if n != 1 else ''}."
    return ""


# ─── INPUT BOX ────────────────────────────────────────────────────────────────

class InputBox:
    PRESETS = [
        "What is retrieval-augmented generation?",
        "What is AutoresearchClaw?",
        "Compare BDI and reactive agent architectures",
    ]

    def __init__(self, rect):
        self.rect = pygame.Rect(rect)
        self.text = self.PRESETS[0]
        self.active = False
        self.cursor_on = True
        self.cursor_t = 0.0

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F1:
                self.text = self.PRESETS[0]
            elif event.key == pygame.K_F2:
                self.text = self.PRESETS[1]
            elif event.key == pygame.K_F3:
                self.text = self.PRESETS[2]
            elif self.active:
                if event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                elif event.key == pygame.K_RETURN:
                    return "submit"
                elif event.unicode and event.unicode.isprintable():
                    if len(self.text) < 200:
                        self.text += event.unicode
        return None

    def update(self, dt):
        self.cursor_t += dt
        if self.cursor_t >= 0.5:
            self.cursor_t = 0
            self.cursor_on = not self.cursor_on

    def draw(self, surf, font, font_tiny):
        # Box
        bg = (35, 35, 55) if not self.active else (45, 45, 70)
        border = ACCENT_GOLD if self.active else PANEL_BORDER
        pygame.draw.rect(surf, bg, self.rect, border_radius=4)
        pygame.draw.rect(surf, border, self.rect, 2, border_radius=4)
        # Text
        display = self.text
        if self.active and self.cursor_on:
            display += "_"
        text_surf = font.render(display, True, TEXT_LIGHT)
        surf.blit(text_surf, (self.rect.x + 12, self.rect.y + (self.rect.h - text_surf.get_height()) // 2))
        # Hint
        hint = "ENTER to run · F1/F2/F3 for presets · click to edit"
        surf.blit(font_tiny.render(hint, True, TEXT_DIM), (self.rect.x, self.rect.y + self.rect.h + 4))


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    pygame.display.set_caption("Multi-Agent Research · Arcade Edition")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("Courier New", 28, bold=True)
    font = pygame.font.SysFont("Courier New", 18, bold=True)
    font_small = pygame.font.SysFont("Courier New", 14)
    font_tiny = pygame.font.SysFont("Courier New", 12)

    # Runtime state
    agents: dict[str, AgentRuntime] = {
        name: AgentRuntime(name=name, style=style) for name, style in AGENTS.items()
    }
    last_active_edges: set[tuple[str, str]] = set()
    pipeline_thread: threading.Thread | None = None
    pipeline_running = False
    pipeline_started_at = 0.0
    final_answer = ""
    final_scroll = 0  # scroll offset for final answer panel

    input_box = InputBox((40, 30, WIDTH - 380, 40))

    running = True
    t = 0.0

    while running:
        dt = clock.tick(FPS) / 1000.0
        t += dt

        # ─── EVENTS ──────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                final_scroll = max(0, final_scroll - event.y * 24)
            else:
                action = input_box.handle(event)
                if action == "submit" and not pipeline_running:
                    # Reset agent states
                    for a in agents.values():
                        a.state = "idle"
                        a.bubble_text = ""
                        a.bubble_until = 0
                        a.fire_count = 0
                    last_active_edges = set()
                    final_answer = ""
                    final_scroll = 0
                    # Start pipeline in background
                    query = input_box.text.strip()
                    if query:
                        pipeline_running = True
                        pipeline_started_at = t
                        pipeline_thread = threading.Thread(
                            target=run_pipeline, args=(query,), daemon=True
                        )
                        pipeline_thread.start()

        # ─── DRAIN PIPELINE EVENTS ───────────────────────────────────────────
        while True:
            try:
                kind, name, payload = EVENT_Q.get_nowait()
            except queue.Empty:
                break
            if kind == "start" and name in agents:
                a = agents[name]
                a.state = "working"
                a.last_event_t = t
                a.fire_count += 1
                # Light up incoming edges ONLY where the source agent has already
                # fired — otherwise the orange critic→researcher loop arrows would
                # light up on the initial researcher fan-out, even though critic
                # hasn't run yet. Gating by source.fire_count fixes the apparent
                # "orange arrows pointing the wrong way" visual.
                for src_, dst, _ in CONNECTIONS:
                    if dst == name and agents[src_].fire_count > 0:
                        last_active_edges.add((src_, dst))
            elif kind == "done" and name in agents:
                a = agents[name]
                a.state = "done"
                a.last_event_t = t
                a.bubble_text = summarize_output(name, payload or {})
                a.bubble_until = t + 999  # keep until next pipeline run
                # Clear incoming arrows for this node
                last_active_edges = {(s, d) for (s, d) in last_active_edges if d != name}
            elif kind == "complete":
                pipeline_running = False
                last_active_edges = set()
                if payload:
                    final_answer = (payload.get("final_answer") or payload.get("synthesis") or "").strip()
            elif kind == "crashed":
                pipeline_running = False
                last_active_edges = set()
                final_answer = f"⚠ Pipeline crashed: {payload}"
            elif kind == "error" and name in agents:
                agents[name].state = "error"
                agents[name].bubble_text = f"Error: {payload}"
                agents[name].bubble_until = t + 999

        # ─── INPUT BOX TICK ──────────────────────────────────────────────────
        input_box.update(dt)

        # ─── RENDER ──────────────────────────────────────────────────────────
        draw_tiled_floor(screen)

        # Title
        title_surf = font_title.render("🎮  MULTI-AGENT RESEARCH  ·  ARCADE", True, ACCENT_GOLD)
        screen.blit(title_surf, ((WIDTH - title_surf.get_width()) // 2, 80))

        # Status indicator
        if pipeline_running:
            elapsed = t - pipeline_started_at
            status = f"●  RUNNING  {elapsed:5.1f}s"
            status_color = (100, 220, 120)
        elif final_answer:
            status = "★  COMPLETE"
            status_color = ACCENT_GOLD
        else:
            status = "○  IDLE"
            status_color = TEXT_DIM
        screen.blit(font.render(status, True, status_color), (WIDTH - 320, 40))

        # Connections (drawn before agents so they appear behind)
        for src_, dst, kind in CONNECTIONS:
            is_active = (src_, dst) in last_active_edges
            base = AGENTS[src_].color if is_active else (70, 70, 100)
            color = base if is_active else (70, 70, 100)
            draw_arrow(
                screen,
                AGENTS[src_].pos,
                AGENTS[dst].pos,
                color,
                dashed=(kind == "loop"),
                animated=is_active,
                t=t,
            )

        # Agent characters + nameplates
        for name, runtime in agents.items():
            style = runtime.style
            cx, cy = style.pos
            draw_character(screen, cx, cy, style, runtime.state, t - runtime.last_event_t)
            # Nameplate
            plate_w = 130
            plate_h = 30
            plate_x = cx - plate_w // 2
            plate_y = cy + 70
            plate_color = style.color if runtime.state != "idle" else (50, 50, 75)
            pygame.draw.rect(screen, plate_color, (plate_x, plate_y, plate_w, plate_h), border_radius=4)
            pygame.draw.rect(screen, (20, 20, 30), (plate_x, plate_y, plate_w, plate_h), 2, border_radius=4)
            name_surf = font.render(style.display, True, (255, 255, 255))
            screen.blit(name_surf, (cx - name_surf.get_width() // 2, plate_y + 6))
            # Fire count badge (if > 1, means it iterated)
            if runtime.fire_count > 1:
                bx, by = plate_x + plate_w - 18, plate_y - 8
                pygame.draw.circle(screen, ACCENT_GOLD, (bx, by), 12)
                pygame.draw.circle(screen, (20, 20, 30), (bx, by), 12, 2)
                num = font_small.render(f"x{runtime.fire_count}", True, (20, 20, 30))
                screen.blit(num, (bx - num.get_width() // 2, by - num.get_height() // 2))

        # Speech bubbles (rendered last so they're on top)
        for runtime in agents.values():
            if runtime.bubble_text and t < runtime.bubble_until:
                draw_speech_bubble(screen, font_small, font_tiny, runtime.style.pos, runtime.bubble_text)

        # Input box
        input_box.draw(screen, font, font_tiny)

        # Final answer panel — centered modal-style with dimmed backdrop
        if final_answer:
            # Semi-transparent overlay over everything except the top input strip
            # so the user can still type a new query to dismiss
            top_strip_h = 100
            dim = pygame.Surface((WIDTH, HEIGHT - top_strip_h), pygame.SRCALPHA)
            dim.fill((10, 10, 20, 180))
            screen.blit(dim, (0, top_strip_h))

            panel_w = int(WIDTH * 0.78)
            panel_h = int(HEIGHT * 0.70)
            panel_x = (WIDTH - panel_w) // 2
            panel_y = (HEIGHT - panel_h) // 2 + 20
            panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)

            # Outer glow
            glow = pygame.Surface((panel_w + 24, panel_h + 24), pygame.SRCALPHA)
            pygame.draw.rect(glow, (255, 215, 0, 60), glow.get_rect(), border_radius=14)
            screen.blit(glow, (panel_x - 12, panel_y - 12))

            pygame.draw.rect(screen, PANEL_BG, panel_rect, border_radius=10)
            pygame.draw.rect(screen, ACCENT_GOLD, panel_rect, 3, border_radius=10)

            # Title
            title = font_title.render("★  FINAL ANSWER  ★", True, ACCENT_GOLD)
            screen.blit(title, (panel_x + (panel_w - title.get_width()) // 2, panel_y + 18))
            # Divider under title
            pygame.draw.line(
                screen, (90, 90, 130),
                (panel_x + 30, panel_y + 60),
                (panel_x + panel_w - 30, panel_y + 60),
                1,
            )

            # Scrollable body
            body_rect = pygame.Rect(panel_x + 30, panel_y + 72, panel_w - 60, panel_h - 116)
            clip = screen.get_clip()
            screen.set_clip(body_rect)
            wrapped_lines = []
            for paragraph in final_answer.split("\n"):
                wrapped_lines.extend(textwrap.wrap(paragraph, width=100) or [""])
            line_h = font_small.get_linesize() + 4  # extra spacing for readability
            for i, line in enumerate(wrapped_lines):
                y = body_rect.y + i * line_h - final_scroll
                if -line_h < y < body_rect.bottom:
                    screen.blit(font_small.render(line, True, TEXT_LIGHT), (body_rect.x, y))
            screen.set_clip(clip)

            # Footer hint
            total_h = len(wrapped_lines) * line_h
            footer_parts = []
            if total_h > body_rect.h:
                footer_parts.append("scroll wheel ↑↓")
            footer_parts.append("type new query above and ENTER to start over")
            hint = font_tiny.render("  ·  ".join(footer_parts), True, TEXT_DIM)
            screen.blit(hint, (panel_x + (panel_w - hint.get_width()) // 2, panel_y + panel_h - 26))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
