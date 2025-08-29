# 🎓 Academic Assistant (Personal Project)

## 📖 Description
**Personalised Academic Assistant using Agentic AI**  
This project is an AI-driven academic assistant that generates **personalised study plans, structured notes, and strategic learning advice**.  
It coordinates multiple agents through a **ReAct-based decision-making loop** using the `gpt-oss-20b` model (or OpenAI GPT-4 alternative) and leverages **LangGraph** for workflow orchestration.

---

## ⚙️ Tech Stack
- **Python 3.9+**
- **gpt-oss-20b** / **OpenAI GPT-4**
- **LangChain / LangGraph** (workflow orchestration & conditional routing)
- **Pydantic** (data validation & typed state management)
- **Rich** (for console-based Markdown, Panels, and styling)
- **Google Colab Integration** (file uploads & secrets handling)
- **AsyncIO** (parallel execution of agents)

---

## ✨ Features
- **Coordinator Agent**: Uses ReACT to orchestrate PLANNER, NOTEWRITER, and ADVISOR agents.
- **Planner Agent**: Builds **personalised, time-optimized study schedules** based on profile, calendar, and tasks.
- **NoteWriter Agent**: Generates **concise study notes** adapted to learning style.
- **Advisor Agent**: Provides **personalised strategic guidance** for time, energy, and stress management.
- **Profile Analyzer**: Extracts **learning style patterns** and productivity factors.
- **LangGraph Workflow**:
  - **Conditional Routing**: Agents are invoked based on request type and context.
  - **Parallel Execution**: Multiple agents can run concurrently.
  - **State Management**: Consolidates agent outputs for robust decision-making.
- **Interactive Console Output**: Plans, notes, and advice are displayed using Rich formatting.

---
