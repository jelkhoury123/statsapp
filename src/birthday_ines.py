"""Birthday celebration for Ines — active through 2026-10-25 (Europe/Paris).

Lightweight in-page overlay; fades via CSS after 3s. No close button.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import streamlit as st

PARIS = ZoneInfo("Europe/Paris")
BIRTHDAY_LAST_DAY = date(2026, 10, 25)
_PLAYED_KEY = "ines_bday_2026_played"
_REPLAY_KEY = "ines_bday_2026_replay"


def is_birthday_window_open(now: datetime | None = None) -> bool:
    moment = now if now is not None else datetime.now(PARIS)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=PARIS)
    else:
        moment = moment.astimezone(PARIS)
    return moment.date() <= BIRTHDAY_LAST_DAY


def _celebration_html() -> str:
    return """
<style>
@keyframes ines-bday-fade {
  0%   { opacity: 0; }
  12%  { opacity: 1; }
  78%  { opacity: 1; }
  100% { opacity: 0; visibility: hidden; pointer-events: none; }
}
@keyframes ines-bday-glow {
  from { filter: drop-shadow(0 0 14px rgba(236, 72, 153, 0.35)); }
  to   { filter: drop-shadow(0 0 26px rgba(236, 72, 153, 0.65)); }
}
@keyframes ines-bday-fall {
  to { transform: translateY(110vh) rotate(520deg); opacity: 0.1; }
}
#ines-bday-overlay {
  position: fixed; inset: 0; z-index: 2147483000;
  display: flex; align-items: center; justify-content: center;
  background: radial-gradient(ellipse at 50% 30%,
    rgba(255, 240, 250, 0.82) 0%,
    rgba(237, 224, 255, 0.78) 45%,
    rgba(15, 23, 42, 0.28) 100%);
  backdrop-filter: blur(2.5px);
  -webkit-backdrop-filter: blur(2.5px);
  pointer-events: none;
  animation: ines-bday-fade 3s ease forwards;
  font-family: "Segoe UI", "Helvetica Neue", system-ui, sans-serif;
}
#ines-bday-overlay .ines-card {
  text-align: center;
  padding: 2.1rem 2.6rem 1.85rem;
  border-radius: 22px;
  background: linear-gradient(165deg, #fffdf9 0%, #fff5fb 48%, #f3e8ff 100%);
  border: 1px solid rgba(212, 175, 120, 0.55);
  box-shadow: 0 24px 60px rgba(88, 28, 135, 0.18);
  max-width: min(420px, 92vw);
  position: relative; z-index: 2;
}
#ines-bday-overlay .ines-sparkle {
  margin: 0 0 0.35rem 0;
  font-size: 0.78rem; font-weight: 700; letter-spacing: 0.28em;
  color: #A16207; text-transform: uppercase;
}
#ines-bday-overlay .ines-name {
  margin: 0.15rem 0 0.1rem 0;
  font-size: clamp(2.4rem, 7vw, 3.35rem);
  font-weight: 780; letter-spacing: 0.12em;
  background: linear-gradient(120deg, #BE185D, #7C3AED 55%, #D97706);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
#ines-bday-overlay .ines-twenty {
  margin: 0.15rem 0 0.55rem 0;
  font-size: clamp(4.2rem, 14vw, 6.4rem);
  font-weight: 860; line-height: 1;
  background: linear-gradient(135deg, #F59E0B 0%, #EC4899 45%, #A78BFA 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: ines-bday-glow 2.4s ease-in-out infinite alternate;
}
#ines-bday-overlay .ines-tagline {
  margin: 0; font-size: 1.02rem; font-weight: 550; color: #6B21A8;
  font-style: italic; line-height: 1.4;
}
#ines-bday-overlay .ines-confetti {
  position: absolute; inset: 0; overflow: hidden; pointer-events: none;
}
#ines-bday-overlay .ines-piece {
  position: absolute; top: -12px; width: 8px; height: 12px;
  border-radius: 2px; opacity: 0.9;
  animation: ines-bday-fall linear forwards;
}
@media (prefers-reduced-motion: reduce) {
  #ines-bday-overlay { animation: ines-bday-fade 2s ease forwards; }
  #ines-bday-overlay .ines-twenty { animation: none !important; }
  #ines-bday-overlay .ines-piece { display: none !important; }
}
.st-key-ines_bday_badge {
  position: fixed !important; top: 0.45rem !important; right: 0.7rem !important;
  z-index: 99990 !important;
}
.st-key-ines_bday_badge button {
  min-width: 2.35rem !important; width: 2.35rem !important;
  height: 2.35rem !important; padding: 0 !important;
  border-radius: 999px !important;
  border: 1px solid rgba(217, 119, 6, 0.45) !important;
  background: linear-gradient(145deg, #FFF7ED, #FCE7F3 55%, #EDE9FE) !important;
  box-shadow: 0 6px 18px rgba(124, 58, 237, 0.22) !important;
  font-size: 1.15rem !important;
}
</style>
<div id="ines-bday-overlay" role="dialog" aria-label="Happy birthday Ines">
  <div class="ines-confetti" aria-hidden="true">
    <span class="ines-piece" style="left:8%;background:#F59E0B;animation-duration:2.8s"></span>
    <span class="ines-piece" style="left:22%;background:#EC4899;animation-duration:3.1s;animation-delay:.2s"></span>
    <span class="ines-piece" style="left:36%;background:#A78BFA;animation-duration:2.6s;animation-delay:.1s"></span>
    <span class="ines-piece" style="left:50%;background:#FDE68A;animation-duration:3.3s;animation-delay:.4s"></span>
    <span class="ines-piece" style="left:64%;background:#F9A8D4;animation-duration:2.9s;animation-delay:.15s"></span>
    <span class="ines-piece" style="left:78%;background:#C4B5FD;animation-duration:3.0s;animation-delay:.3s"></span>
    <span class="ines-piece" style="left:90%;background:#F59E0B;animation-duration:2.7s;animation-delay:.05s"></span>
  </div>
  <div class="ines-card">
    <p class="ines-sparkle">✨ Happy Birthday ✨</p>
    <p class="ines-name">INES</p>
    <p class="ines-twenty">20</p>
    <p class="ines-tagline">Here's to a brilliant new decade!</p>
  </div>
</div>
"""


def mount_birthday_surprise() -> None:
    if not is_birthday_window_open():
        return
    if st.session_state.pop(_REPLAY_KEY, False):
        st.session_state[_PLAYED_KEY] = False
    if not st.session_state.get(_PLAYED_KEY, False):
        st.session_state[_PLAYED_KEY] = True
        st.markdown(_celebration_html(), unsafe_allow_html=True)
    if st.button("🎂", key="ines_bday_badge", help="Replay birthday greeting"):
        st.session_state[_REPLAY_KEY] = True
        st.rerun()
