"""Clickable student table. A row click toggles that student ID."""

from __future__ import annotations

from functools import lru_cache

import streamlit as st

_CSS = """
:host {
  display: block;
  height: 100%;
  min-height: 0;
  overflow: hidden;
}
.student-scroll {
  height: 100%;
  min-height: 0;
  overflow: auto;
}
table {
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;
  font-family: "Source Sans 3", "Source Sans Pro", sans-serif;
  font-size: 0.82rem;
  color: #1f2937;
}
th, td {
  width: 33.33%;
  padding: 0.28rem 0.45rem;
  text-align: right;
  border-bottom: 1px solid #e5e7eb;
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}
th:first-child, td:first-child { text-align: left; }
thead th {
  position: sticky;
  top: 0;
  background: #f8fafc;
  font-weight: 600;
  z-index: 1;
}
tbody tr { cursor: pointer; }
tbody tr:hover { background: #f3f4f6; }
tbody tr.selected { background: #dbeafe; }
tbody tr.selected:hover { background: #bfdbfe; }
"""

_JS = """
export default function(component) {
  const { data, setTriggerValue, parentElement } = component;
  const rows = (data && data.rows) || [];
  const selected = new Set(((data && data.selected) || []).map(Number));
  const token = data && data.token != null ? String(data.token) : "";
  const focus = data && data.focus != null ? Number(data.focus) : null;

  let scroller = parentElement.querySelector(".student-scroll");
  if (!scroller) {
    scroller = document.createElement("div");
    scroller.className = "student-scroll";
    parentElement.appendChild(scroller);
  }
  const previousScroll = scroller.scrollTop;
  const samePage = scroller.dataset.token === token;
  scroller.replaceChildren();

  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const head = document.createElement("tr");
  [
    ["ID", "Student ID"],
    ["Hours", "Hours studied"],
    ["Score", "Exam score"],
  ].forEach(([label, tip]) => {
    const cell = document.createElement("th");
    cell.textContent = label;
    cell.title = tip;
    head.appendChild(cell);
  });
  thead.appendChild(head);
  table.appendChild(thead);

  const body = document.createElement("tbody");
  let focusRow = null;
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    const studentId = Number(row.id);
    tr.dataset.id = String(studentId);
    if (selected.has(studentId)) tr.classList.add("selected");
    ["id", "hours", "score"].forEach((key) => {
      const cell = document.createElement("td");
      cell.textContent = String(row[key]);
      tr.appendChild(cell);
    });
    tr.addEventListener("click", () => setTriggerValue("pick", studentId));
    if (focus !== null && studentId === focus) focusRow = tr;
    body.appendChild(tr);
  });
  table.appendChild(body);
  scroller.appendChild(table);
  scroller.dataset.token = token;

  if (focusRow) {
    focusRow.scrollIntoView({ block: "nearest" });
  } else if (samePage) {
    scroller.scrollTop = previousScroll;
  }
}
"""


@lru_cache(maxsize=1)
def student_table():
    """Register the table once. Later script reruns reuse the same component."""
    return st.components.v2.component(
        "intro_student_table",
        css=_CSS,
        js=_JS,
        isolate_styles=True,
    )
