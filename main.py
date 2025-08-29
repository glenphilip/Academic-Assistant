import operator
from functools import reduce
from typing import Annotated, List, Dict, TypedDict, Literal, Optional, Callable, Set, Tuple, Any, Union, TypeVar
from datetime import datetime, timezone, timedelta
import asyncio
from pydantic import BaseModel, Field
from operator import add
from IPython.display import Image, display
from google.colab import files
import json
import re
import os
from openai import OpenAI, AsyncOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, Graph, END, START
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.style import Style

OPENAI_API_KEY = None

def configure_api_keys():
    from google.colab import userdata
    global OPENAI_API_KEY
    try:
        # Try to get from Google Colab secrets first
        OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
    except:
        # Fallback to environment variable
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    if OPENAI_API_KEY:
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
        
    is_configured = bool(OPENAI_API_KEY)
    print(f"API Key configured: {is_configured}")
    return is_configured

api_configured = configure_api_keys()
if not api_configured:
    print("\nAPI key not found. Please ensure you have:")
    print("1. Set up your API key in Google Colab secrets as 'OPENAI_API_KEY', or")
    print("2. Set OPENAI_API_KEY environment variable")

T = TypeVar('T')

def dict_reducer(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = dict_reducer(merged[key], value)
        else:
            merged[key] = value
    return merged

class AcademicState(TypedDict):
    messages: Annotated[List[BaseMessage], add]
    profile: Annotated[Dict, dict_reducer]
    calendar: Annotated[Dict, dict_reducer]
    tasks: Annotated[Dict, dict_reducer]
    results: Annotated[Dict[str, Any], dict_reducer]

class LLMConfig:
    base_url: str = "https://api.openai.com/v1"  # Changed to OpenAI endpoint
    model: str = "gpt-4"  # Changed to GPT-4 model
    max_tokens: int = 1024
    default_temp: float = 0.5

class OpenAILLM:  # Renamed from NeMoLLaMa
    def __init__(self, api_key: str):
        self.config = LLMConfig()
        self.client = AsyncOpenAI(
            api_key=api_key  # OpenAI client doesn't need base_url parameter for standard endpoint
        )
        self._is_authenticated = False

    async def check_auth(self) -> bool:
        test_message = [{"role": "user", "content": "test"}]
        try:
            await self.agenerate(test_message, temperature=0.1)
            self._is_authenticated = True
            return True
        except Exception as e:
            print(f"❌ Authentication failed: {str(e)}")
            return False

    async def agenerate(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None
    ) -> str:
        completion = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature or self.config.default_temp,
            max_tokens=self.config.max_tokens,
            stream=False
        )
        return completion.choices[0].message.content

class DataManager:
    def __init__(self):
        self.profile_data = None
        self.calendar_data = None
        self.task_data = None

    def load_data(self, profile_json: str, calendar_json: str, task_json: str):
        self.profile_data = json.loads(profile_json)
        self.calendar_data = json.loads(calendar_json)
        self.task_data = json.loads(task_json)

    def get_student_profile(self, student_id: str) -> Dict:
        if self.profile_data:
            return next((p for p in self.profile_data["profiles"]
                        if p["id"] == student_id), None)
        return None

    def parse_datetime(self, dt_str: str) -> datetime:
        try:
            dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            return dt.astimezone(timezone.utc)
        except ValueError:
            dt = datetime.fromisoformat(dt_str)
            return dt.replace(tzinfo=timezone.utc)

    def get_upcoming_events(self, days: int = 7) -> List[Dict]:
        if not self.calendar_data:
            return []

        now = datetime.now(timezone.utc)
        future = now + timedelta(days=days)

        events = []
        for event in self.calendar_data.get("events", []):
            try:
                start_time = self.parse_datetime(event["start"]["dateTime"])
                if now <= start_time <= future:
                    events.append(event)
            except (KeyError, ValueError) as e:
                print(f"Warning: Could not process event due to {str(e)}")
                continue

        return events

    def get_active_tasks(self) -> List[Dict]:
        if not self.task_data:
            return []

        now = datetime.now(timezone.utc)
        active_tasks = []

        for task in self.task_data.get("tasks", []):
            try:
                due_date = self.parse_datetime(task["due"])
                if task["status"] == "needsAction" and due_date > now:
                    task["due_datetime"] = due_date
                    active_tasks.append(task)
            except (KeyError, ValueError) as e:
                print(f"Warning: Could not process task due to {str(e)}")
                continue

        return active_tasks

class AgentExecutor:
    def __init__(self, llm):
        self.llm = llm
        self.agents = {
            "PLANNER": PlannerAgent(llm),
            "NOTEWRITER": NoteWriterAgent(llm),
            "ADVISOR": AdvisorAgent(llm)
        }

    async def execute(self, state: AcademicState) -> Dict:
        try:
            analysis = state["results"].get("coordinator_analysis", {})
            required_agents = analysis.get("required_agents", ["PLANNER"])
            concurrent_groups = analysis.get("concurrent_groups", [])
            results = {}

            for group in concurrent_groups:
                tasks = []
                for agent_name in group:
                    if agent_name in required_agents and agent_name in self.agents:
                        tasks.append(self.agents[agent_name](state))

                if tasks:
                    group_results = await asyncio.gather(*tasks, return_exceptions=True)
                    for agent_name, result in zip(group, group_results):
                        if not isinstance(result, Exception):
                            results[agent_name.lower()] = result

            if not results and "PLANNER" in self.agents:
                planner_result = await self.agents["PLANNER"](state)
                results["planner"] = planner_result

            print("agent_outputs", results)

            return {
                "results": {
                    "agent_outputs": results
                }
            }

        except Exception as e:
            print(f"Execution error: {e}")
            return {
                "results": {
                    "agent_outputs": {
                        "planner": {
                            "plan": "Emergency fallback plan: Please try again or contact support."
                        }
                    }
                }
            }

class AgentAction(BaseModel):
    action: str
    thought: str
    tool: Optional[str] = None
    action_input: Optional[Dict] = None

class AgentOutput(BaseModel):
    observation: str
    output: Dict

class ReActAgent:
    def __init__(self, llm):
        self.llm = llm
        self.few_shot_examples = []
        self.tools = {
            "search_calendar": self.search_calendar,
            "analyze_tasks": self.analyze_tasks,
            "check_learning_style": self.check_learning_style,
            "check_performance": self.check_performance
        }

    async def search_calendar(self, state: AcademicState) -> List[Dict]:
        events = state["calendar"].get("events", [])
        now = datetime.now(timezone.utc)
        return [e for e in events if datetime.fromisoformat(e["start"]["dateTime"]) > now]

    async def analyze_tasks(self, state: AcademicState) -> List[Dict]:
        return state["tasks"].get("tasks", [])

    async def check_learning_style(self, state: AcademicState) -> AcademicState:
        profile = state["profile"]
        learning_data = {
            "style": profile.get("learning_preferences", {}).get("learning_style", {}),
            "patterns": profile.get("learning_preferences", {}).get("study_patterns", {})
        }
        if "results" not in state:
            state["results"] = {}
        state["results"]["learning_analysis"] = learning_data
        return state

    async def check_performance(self, state: AcademicState) -> AcademicState:
        profile = state["profile"]
        courses = profile.get("academic_info", {}).get("current_courses", [])
        if "results" not in state:
            state["results"] = {}
        state["results"]["performance_analysis"] = {"courses": courses}
        return state

async def analyze_context(state: AcademicState) -> Dict:
    profile = state.get("profile", {})
    calendar = state.get("calendar", {})
    tasks = state.get("tasks", {})

    courses = profile.get("academic_info", {}).get("current_courses", [])
    current_course = None
    request = state["messages"][-1].content.lower()

    for course in courses:
        if course["name"].lower() in request:
            current_course = course
            break

    return {
        "student": {
            "major": profile.get("personal_info", {}).get("major", "Unknown"),
            "year": profile.get("personal_info", {}).get("academic_year"),
            "learning_style": profile.get("learning_preferences", {}).get("learning_style", {}),
        },
        "course": current_course,
        "upcoming_events": len(calendar.get("events", [])),
        "active_tasks": len(tasks.get("tasks", [])),
        "study_patterns": profile.get("learning_preferences", {}).get("study_patterns", {})
    }

def parse_coordinator_response(response: str) -> Dict:
    try:
        analysis = {
            "required_agents": ["PLANNER"],
            "priority": {"PLANNER": 1},
            "concurrent_groups": [["PLANNER"]],
            "reasoning": "Default coordination"
        }

        if "Thought:" in response and "Decision:" in response:
            if "NoteWriter" in response or "note" in response.lower():
                analysis["required_agents"].append("NOTEWRITER")
                analysis["priority"]["NOTEWRITER"] = 2
                analysis["concurrent_groups"] = [["PLANNER", "NOTEWRITER"]]

            if "Advisor" in response or "guidance" in response.lower():
                analysis["required_agents"].append("ADVISOR")
                analysis["priority"]["ADVISOR"] = 3

            thought_section = response.split("Thought:")[1].split("Action:")[0].strip()
            analysis["reasoning"] = thought_section

        return analysis

    except Exception as e:
        print(f"Parse error: {str(e)}")
        return {
            "required_agents": ["PLANNER"],
            "priority": {"PLANNER": 1},
            "concurrent_groups": [["PLANNER"]],
            "reasoning": "Fallback due to parse error"
        }

COORDINATOR_PROMPT = """You are a Coordinator Agent using ReACT framework to orchestrate multiple academic support agents.

AVAILABLE AGENTS:
• PLANNER: Handles scheduling and time management
• NOTEWRITER: Creates study materials and content summaries
• ADVISOR: Provides personalized academic guidance

PARALLEL EXECUTION RULES:
1. Group compatible agents that can run concurrently
2. Maintain dependencies between agent executions
3. Coordinate results from parallel executions

REACT PATTERN:
Thought: [Analyze request complexity and required support types]
Action: [Select optimal agent combination]
Observation: [Evaluate selected agents' capabilities]
Decision: [Finalize agent deployment plan]

ANALYSIS POINTS:
1. Task Complexity and Scope
2. Time Constraints
3. Resource Requirements
4. Learning Style Alignment
5. Support Type Needed

CONTEXT:
Request: {request}
Student Context: {context}

FORMAT RESPONSE AS:
Thought: [Analysis of academic needs and context]
Action: [Agent selection and grouping strategy]
Observation: [Expected workflow and dependencies]
Decision: [Final agent deployment plan with rationale]
"""

async def coordinator_agent(state: AcademicState) -> Dict:
    try:
        context = await analyze_context(state)
        query = state["messages"][-1].content

        prompt = COORDINATOR_PROMPT

        response = await llm.agenerate([
            {"role": "system", "content": prompt.format(
                request=query,
                context=json.dumps(context, indent=2)
            )}
        ])

        analysis = parse_coordinator_response(response)
        return {
            "results": {
                "coordinator_analysis": {
                    "required_agents": analysis.get("required_agents", ["PLANNER"]),
                    "priority": analysis.get("priority", {"PLANNER": 1}),
                    "concurrent_groups": analysis.get("concurrent_groups", [["PLANNER"]]),
                    "reasoning": response
                }
            }
        }

    except Exception as e:
        print(f"Coordinator error: {e}")
        return {
            "results": {
                "coordinator_analysis": {
                    "required_agents": ["PLANNER"],
                    "priority": {"PLANNER": 1},
                    "concurrent_groups": [["PLANNER"]],
                    "reasoning": "Error in coordination. Falling back to planner."
                }
            }
        }

PROFILE_ANALYZER_PROMPT = """You are a Profile Analysis Agent using the ReACT framework to analyze student profiles.

OBJECTIVE:
Analyze the student profile and extract key learning patterns that will impact their academic success.

REACT PATTERN:
Thought: Analyze what aspects of the profile need investigation
Action: Extract specific information from relevant profile sections
Observation: Note key patterns and implications
Response: Provide structured analysis

PROFILE DATA:
{profile}

ANALYSIS FRAMEWORK:
1. Learning Characteristics:
    • Primary learning style
    • Information processing patterns
    • Attention span characteristics

2. Environmental Factors:
    • Optimal study environment
    • Distraction triggers
    • Productive time periods

3. Executive Function:
    • Task management patterns
    • Focus duration limits
    • Break requirements

4. Energy Management:
    • Peak energy periods
    • Recovery patterns
    • Fatigue signals

INSTRUCTIONS:
1. Use the ReACT pattern for each analysis area
2. Provide specific, actionable observations
3. Note both strengths and challenges
4. Identify patterns that affect study planning

FORMAT YOUR RESPONSE AS:
Thought: [Initial analysis of profile components]
Action: [Specific areas being examined]
Observation: [Patterns and insights discovered]
Analysis Summary: [Structured breakdown of key findings]
Recommendations: [Specific adaptations needed]
"""

async def profile_analyzer(state: AcademicState) -> Dict:
    profile = state["profile"]
    prompt = PROFILE_ANALYZER_PROMPT

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(profile)}
    ]

    response = await llm.agenerate(messages)

    return {
        "results": {
            "profile_analysis": {
                "analysis": response
            }
        }
    }

class PlannerAgent(ReActAgent):
    def __init__(self, llm):
        super().__init__(llm)
        self.llm = llm
        self.few_shot_examples = self._initialize_fewshots()
        self.workflow = self.create_subgraph()

    def _initialize_fewshots(self):
        return [
            {
                "input": "Help with exam prep while managing ADHD and football",
                "thought": "Need to check calendar conflicts and energy patterns",
                "action": "search_calendar",
                "observation": "Football match at 6PM, exam tomorrow 9AM",
                "plan": """ADHD-OPTIMIZED SCHEDULE:
                PRE-FOOTBALL (2PM-5PM):
                - 3x20min study sprints
                - Movement breaks
                - Quick rewards after each sprint

                FOOTBALL MATCH (6PM-8PM):
                - Use as dopamine reset
                - Formula review during breaks

                POST-MATCH (9PM-12AM):
                - Environment: Café noise
                - 15/5 study/break cycles
                - Location changes hourly

                EMERGENCY PROTOCOLS:
                - Focus lost → jumping jacks
                - Overwhelmed → room change
                - Brain fog → cold shower"""
            },
            {
                "input": "Struggling with multiple deadlines",
                "thought": "Check task priorities and performance issues",
                "action": "analyze_tasks",
                "observation": "3 assignments due, lowest grade in Calculus",
                "plan": """PRIORITY SCHEDULE:
                HIGH-FOCUS SLOTS:
                - Morning: Calculus practice
                - Post-workout: Assignments
                - Night: Quick reviews

                ADHD MANAGEMENT:
                - Task timer challenges
                - Reward system per completion
                - Study buddy accountability"""
            }
        ]

    def create_subgraph(self) -> StateGraph:
        subgraph = StateGraph(AcademicState)
        subgraph.add_node("calendar_analyzer", self.calendar_analyzer)
        subgraph.add_node("task_analyzer", self.task_analyzer)
        subgraph.add_node("plan_generator", self.plan_generator)
        subgraph.add_edge("calendar_analyzer", "task_analyzer")
        subgraph.add_edge("task_analyzer", "plan_generator")
        subgraph.set_entry_point("calendar_analyzer")
        return subgraph.compile()

    async def calendar_analyzer(self, state: AcademicState) -> AcademicState:
        events = state["calendar"].get("events", [])
        now = datetime.now(timezone.utc)
        future = now + timedelta(days=7)

        filtered_events = [
            event for event in events
            if now <= datetime.fromisoformat(event["start"]["dateTime"]) <= future
        ]

        prompt = """Analyze calendar events and identify:
        Events: {events}

        Focus on:
        - Available time blocks
        - Energy impact of activities
        - Potential conflicts
        - Recovery periods
        - Study opportunity windows
        - Activity patterns
        - Schedule optimization
        """

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(filtered_events)}
        ]

        response = await self.llm.agenerate(messages)

        return {
            "results": {
                "calendar_analysis": {
                    "analysis": response
                }
            }
        }

    async def task_analyzer(self, state: AcademicState) -> AcademicState:
        tasks = state["tasks"].get("tasks", [])

        prompt = """Analyze tasks and create priority structure:
        Tasks: {tasks}

        Consider:
        - Urgency levels
        - Task complexity
        - Energy requirements
        - Dependencies
        - Required focus levels
        - Time estimations
        - Learning objectives
        - Success criteria
        """

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(tasks)}
        ]

        response = await self.llm.agenerate(messages)

        return {
            "results": {
                "task_analysis": {
                    "analysis": response
                }
            }
        }

    async def plan_generator(self, state: AcademicState) -> AcademicState:
        profile_analysis = state["results"]["profile_analysis"]
        calendar_analysis = state["results"]["calendar_analysis"]
        task_analysis = state["results"]["task_analysis"]

        prompt = f"""AI Planning Assistant: Create focused study plan using ReACT framework.

INPUT CONTEXT:
- Profile Analysis: {profile_analysis}
- Calendar Analysis: {calendar_analysis}
- Task Analysis: {task_analysis}

EXAMPLES:
{json.dumps(self.few_shot_examples, indent=2)}

INSTRUCTIONS:
1. Follow ReACT pattern:
  Thought: Analyze situation and needs
  Action: Consider all analyses
  Observation: Synthesize findings
  Plan: Create structured plan

2. Address:
  - ADHD management strategies
  - Energy level optimization
  - Task chunking methods
  - Focus period scheduling
  - Environment switching tactics
  - Recovery period planning
  - Social/sport activity balance

3. Include:
  - Emergency protocols
  - Backup strategies
  - Quick wins
  - Reward system
  - Progress tracking
  - Adjustment triggers

Pls act as an intelligent tool to help the students reach their goals or overcome struggles and answer with informal words.

FORMAT:
Thought: [reasoning and situation analysis]
Action: [synthesis approach]
Observation: [key findings]
Plan: [actionable steps and structural schedule]
"""

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": state["messages"][-1].content}
        ]

        response = await self.llm.agenerate(messages, temperature=0.5)

        return {
            "results": {
                "final_plan": {
                    "plan": response
                }
            }
        }

    async def __call__(self, state: AcademicState) -> Dict:
        try:
            final_state = await self.workflow.ainvoke(state)
            notes = final_state["results"].get("generated_notes", {})
            return {"notes": final_state["results"].get("generated_notes")}
        except Exception as e:
            return {"notes": "Error generating notes. Please try again."}

class NoteWriterAgent(ReActAgent):
    def __init__(self, llm):
        super().__init__(llm)
        self.llm = llm
        self.few_shot_examples = [
            {
                "input": "Need to cram Calculus III for tomorrow",
                "template": "Quick Review",
                "notes": """CALCULUS III ESSENTIALS:

                1. CORE CONCEPTS (80/20 Rule):
                   • Multiple Integrals → volume/area
                   • Vector Calculus → flow/force/rotation
                   • KEY FORMULAS:
                     - Triple integrals in cylindrical/spherical coords
                     - Curl, divergence, gradient relationships

                2. COMMON EXAM PATTERNS:
                   • Find critical points
                   • Calculate flux/work
                   • Optimize with constraints

                3. QUICKSTART GUIDE:
                   • Always draw 3D diagrams
                   • Check units match
                   • Use symmetry to simplify

                4. EMERGENCY TIPS:
                   • If stuck, try converting coordinates
                   • Check boundary conditions
                   • Look for special patterns"""
            }
        ]
        self.workflow = self.create_subgraph()

    def create_subgraph(self) -> StateGraph:
        subgraph = StateGraph(AcademicState)
        subgraph.add_node("notewriter_analyze", self.analyze_learning_style)
        subgraph.add_node("notewriter_generate", self.generate_notes)
        subgraph.add_edge(START, "notewriter_analyze")
        subgraph.add_edge("notewriter_analyze", "notewriter_generate")
        subgraph.add_edge("notewriter_generate", END)
        return subgraph.compile()

    async def analyze_learning_style(self, state: AcademicState) -> AcademicState:
        profile = state["profile"]
        learning_style = profile["learning_preferences"]["learning_style"]

        prompt = f"""Analyze content requirements and determine optimal note structure:

        STUDENT PROFILE:
        - Learning Style: {json.dumps(learning_style, indent=2)}
        - Request: {state['messages'][-1].content}

        FORMAT:
        1. Key Topics (80/20 principle)
        2. Learning Style Adaptations
        3. Time Management Strategy
        4. Quick Reference Format

        FOCUS ON:
        - Essential concepts that give maximum understanding
        - Visual and interactive elements
        - Time-optimized study methods
        """

        response = await self.llm.agenerate([
            {"role": "system", "content": prompt}
        ])

        return {
            "results": {
                "learning_analysis": {
                    "analysis": response
                }
            }
        }

    async def generate_notes(self, state: AcademicState) -> AcademicState:
        analysis = state["results"].get("learning_analysis", "")
        learning_style = state["profile"]["learning_preferences"]["learning_style"]

        prompt = f"""Create concise, high-impact study materials based on analysis:

        ANALYSIS: {analysis}
        LEARNING STYLE: {json.dumps(learning_style, indent=2)}
        REQUEST: {state['messages'][-1].content}

        EXAMPLES:
        {json.dumps(self.few_shot_examples, indent=2)}

        FORMAT:
        **THREE-WEEK INTENSIVE STUDY PLANNER**

        [Generate structured notes with:]
        1. Weekly breakdown
        2. Daily focus areas
        3. Core concepts
        4. Emergency tips
        """

        response = await self.llm.agenerate([
            {"role": "system", "content": prompt}
        ])

        return {
            "results": {
                "generated_notes": {
                    "notes": response
                }
            }
        }

    async def __call__(self, state: AcademicState) -> Dict:
        try:
            final_state = await self.workflow.ainvoke(state)
            notes = final_state["results"].get("generated_notes", {})
            return {"notes": final_state["results"].get("generated_notes")}
        except Exception as e:
            return {"notes": "Error generating notes. Please try again."}

class AdvisorAgent(ReActAgent):
    def __init__(self, llm):
        super().__init__(llm)
        self.llm = llm
        self.few_shot_examples = [
            {
                "request": "Managing multiple deadlines with limited time",
                "profile": {
                    "learning_style": "visual",
                    "workload": "heavy",
                    "time_constraints": ["2 hackathons", "project", "exam"]
                },
                "advice": """PRIORITY-BASED SCHEDULE:

                1. IMMEDIATE ACTIONS
                   • Create visual timeline of all deadlines
                   • Break each task into 45-min chunks
                   • Schedule high-focus work in mornings

                2. WORKLOAD MANAGEMENT
                   • Hackathons: Form team early, set clear roles
                   • Project: Daily 2-hour focused sessions
                   • Exam: Interleaved practice with breaks

                3. ENERGY OPTIMIZATION
                   • Use Pomodoro (25/5) for intensive tasks
                   • Physical activity between study blocks
                   • Regular progress tracking

                4. EMERGENCY PROTOCOLS
                   • If overwhelmed: Take 10min reset break
                   • If stuck: Switch tasks or environments
                   • If tired: Quick power nap, then review"""
            }
        ]
        self.workflow = self.create_subgraph()

    def create_subgraph(self) -> StateGraph:
        subgraph = StateGraph(AcademicState)
        subgraph.add_node("advisor_analyze", self.analyze_situation)
        subgraph.add_node("advisor_generate", self.generate_guidance)
        subgraph.add_edge(START, "advisor_analyze")
        subgraph.add_edge("advisor_analyze", "advisor_generate")
        subgraph.add_edge("advisor_generate", END)
        return subgraph.compile()

    async def analyze_situation(self, state: AcademicState) -> AcademicState:
        profile = state["profile"]
        learning_prefs = profile.get("learning_preferences", {})

        prompt = f"""Analyze student situation and determine guidance approach:

        CONTEXT:
        - Profile: {json.dumps(profile, indent=2)}
        - Learning Preferences: {json.dumps(learning_prefs, indent=2)}
        - Request: {state['messages'][-1].content}

        ANALYZE:
        1. Current challenges
        2. Learning style compatibility
        3. Time management needs
        4. Stress management requirements
        """

        response = await self.llm.agenerate([
            {"role": "system", "content": prompt}
        ])

        return {
            "results": {
                "situation_analysis": {
                    "analysis": response
                }
            }
        }

    async def generate_guidance(self, state: AcademicState) -> AcademicState:
        analysis = state["results"].get("situation_analysis", "")

        prompt = f"""Generate personalized academic guidance based on analysis:

        ANALYSIS: {analysis}
        EXAMPLES: {json.dumps(self.few_shot_examples, indent=2)}

        FORMAT:
        1. Immediate Action Steps
        2. Schedule Optimization
        3. Energy Management
        4. Support Strategies
        5. Emergency Protocols
        """

        response = await self.llm.agenerate([
            {"role": "system", "content": prompt}
        ])

        return {
            "results": {
                "guidance": {
                    "advice": response
                }
            }
        }

    async def __call__(self, state: AcademicState) -> Dict:
        try:
            final_state = await self.workflow.ainvoke(state)
            return {
                "advisor_output": {
                    "guidance": final_state["results"].get("guidance"),
                    "metadata": {
                        "course_specific": True,
                        "considers_learning_style": True
                    }
                }
            }
        except Exception as e:
            return {
                "advisor_output": {
                    "guidance": "Error generating guidance. Please try again."
                }
            }

def create_agents_graph(llm) -> StateGraph:
    workflow = StateGraph(AcademicState)

    planner_agent = PlannerAgent(llm)
    notewriter_agent = NoteWriterAgent(llm)
    advisor_agent = AdvisorAgent(llm)
    executor = AgentExecutor(llm)

    workflow.add_node("coordinator", coordinator_agent)
    workflow.add_node("profile_analyzer", profile_analyzer)
    workflow.add_node("execute", executor.execute)

    def route_to_parallel_agents(state: AcademicState) -> List[str]:
        analysis = state["results"].get("coordinator_analysis", {})
        required_agents = analysis.get("required_agents", [])
        next_nodes = []

        if "PLANNER" in required_agents:
            next_nodes.append("calendar_analyzer")
        if "NOTEWRITER" in required_agents:
            next_nodes.append("notewriter_analyze")
        if "ADVISOR" in required_agents:
            next_nodes.append("advisor_analyze")

        return next_nodes if next_nodes else ["calendar_analyzer"]

    workflow.add_node("calendar_analyzer", planner_agent.calendar_analyzer)
    workflow.add_node("task_analyzer", planner_agent.task_analyzer)
    workflow.add_node("plan_generator", planner_agent.plan_generator)

    workflow.add_node("notewriter_analyze", notewriter_agent.analyze_learning_style)
    workflow.add_node("notewriter_generate", notewriter_agent.generate_notes)

    workflow.add_node("advisor_analyze", advisor_agent.analyze_situation)
    workflow.add_node("advisor_generate", advisor_agent.generate_guidance)

    workflow.add_edge(START, "coordinator")
    workflow.add_edge("coordinator", "profile_analyzer")

    workflow.add_conditional_edges(
        "profile_analyzer",
        route_to_parallel_agents,
        ["calendar_analyzer", "notewriter_analyze", "advisor_analyze"]
    )

    workflow.add_edge("calendar_analyzer", "task_analyzer")
    workflow.add_edge("task_analyzer", "plan_generator")
    workflow.add_edge("plan_generator", "execute")

    workflow.add_edge("notewriter_analyze", "notewriter_generate")
    workflow.add_edge("notewriter_generate", "execute")

    workflow.add_edge("advisor_analyze", "advisor_generate")
    workflow.add_edge("advisor_generate", "execute")

    def should_end(state) -> Union[Literal["coordinator"], Literal[END]]:
        analysis = state["results"].get("coordinator_analysis", {})
        executed = set(state["results"].get("agent_outputs", {}).keys())
        required = set(a.lower() for a in analysis.get("required_agents", []))
        return END if required.issubset(executed) else "coordinator"

    workflow.add_conditional_edges(
        "execute",
        should_end,
        {
            "coordinator": "coordinator",
            END: END
        }
    )

    return workflow.compile()

async def run_all_system(profile_json: str, calendar_json: str, task_json: str):
    try:
        console = Console()
        console.print("\n[bold magenta]🎓 ATLAS: Academic Task Learning Agent System[/bold magenta]")
        console.print("[italic blue]Initializing academic support system with OpenAI GPT...[/italic blue]\n")

        llm = OpenAILLM(OPENAI_API_KEY)  # Changed from NeMoLLaMa
        dm = DataManager()
        dm.load_data(profile_json, calendar_json, task_json)

        # Check authentication
        console.print("[bold blue]Checking OpenAI API authentication...[/bold blue]")
        if await llm.check_auth():
            console.print("[bold green]✓ OpenAI API authentication successful![/bold green]")
        else:
            console.print("[bold red]✗ OpenAI API authentication failed![/bold red]")
            return None, None

        console.print("[bold green]Please enter your academic request:[/bold green]")
        user_input = str(input())
        console.print(f"\n[dim italic]Processing request: {user_input}[/dim italic]\n")

        state = {
            "messages": [HumanMessage(content=user_input)],
            "profile": dm.get_student_profile("student_123"),
            "calendar": {"events": dm.get_upcoming_events()},
            "tasks": {"tasks": dm.get_active_tasks()},
            "results": {}
        }

        graph = create_agents_graph(llm)

        console.print("[bold cyan]System initialized and processing request...[/bold cyan]\n")
        console.print("[bold cyan]Workflow Graph Structure:[/bold cyan]\n")
        display(Image(graph.get_graph().draw_mermaid_png()))

        coordinator_output = None
        final_state = None

        with console.status("[bold green]Processing...", spinner="dots") as status:
            async for step in graph.astream(state):
                if "coordinator_analysis" in step.get("results", {}):
                    coordinator_output = step
                    analysis = coordinator_output["results"]["coordinator_analysis"]

                    console.print("\n[bold cyan]Selected Agents:[/bold cyan]")
                    for agent in analysis.get("required_agents", []):
                        console.print(f"• {agent}")

                if "execute" in step:
                    final_state = step

        if final_state:
            agent_outputs = final_state.get("execute", {}).get("results", {}).get("agent_outputs", {})

            for agent, output in agent_outputs.items():
                console.print(f"\n[bold cyan]{agent.upper()} Output:[/bold cyan]")

                if isinstance(output, dict):
                    for key, value in output.items():
                        if isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                if subvalue and isinstance(subvalue, str):
                                    console.print(subvalue.strip())
                        elif value and isinstance(value, str):
                            console.print(value.strip())
                elif isinstance(output, str):
                    console.print(output.strip())

        console.print("\n[bold green]✓[/bold green] [bold]Task completed![/bold]")
        return coordinator_output, final_state

    except Exception as e:
        console.print(f"\n[bold red]System error:[/bold red] {str(e)}")
        console.print("[yellow]Stack trace:[/yellow]")
        import traceback
        console.print(traceback.format_exc())
        return None, None

async def load_json_and_test():
    print("Academic Assistant Test Setup with OpenAI GPT")
    print("-" * 50)
    print("\nPlease upload your JSON files...")

    try:
        uploaded = files.upload()
        if not uploaded:
            print("No files were uploaded.")
            return

        patterns = {
            'profile': r'profile.*\.json$',
            'calendar': r'calendar.*\.json$',
            'task': r'task.*\.json$'
        }

        found_files = {
            file_type: next((
                f for f in uploaded.keys()
                if re.match(pattern, f, re.IGNORECASE)
            ), None)
            for file_type, pattern in patterns.items()
        }

        missing = [k for k, v in found_files.items() if v is None]
        if missing:
            print(f"Error: Missing required files: {missing}")
            print(f"Uploaded files: {list(uploaded.keys())}")
            return

        print("\nFiles found:")
        for file_type, filename in found_files.items():
            print(f"- {file_type}: {filename}")

        json_contents = {}
        for file_type, filename in found_files.items():
            with open(filename, 'r', encoding='utf-8') as f:
                try:
                    json_contents[file_type] = f.read()
                except Exception as e:
                    print(f"Error reading {file_type} file: {str(e)}")
                    return

        print("\nStarting academic assistance workflow with OpenAI GPT...")
        coordinator_output, output = await run_all_system(
            json_contents['profile'],
            json_contents['calendar'],
            json_contents['task']
        )
        return coordinator_output, output

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nDetailed error information:")
        import traceback
        print(traceback.format_exc())
        return None, None

# Initialize with OpenAI
if OPENAI_API_KEY:
    llm = OpenAILLM(OPENAI_API_KEY)
    data_manager = DataManager()
    print("System initialized with OpenAI GPT-4")
    print("LLM instance:", llm)
    
    # Run the system
    coordinator_output, output = await load_json_and_test()

    # Format and display output
    try:
        if isinstance(output, str):
            json_content = json.loads(output)
        else:
            json_content = output

        plan_content = json_content.get('plan', '')
        plan_content = plan_content.replace('\\n', '\n')
        plan_content = plan_content.replace('\\', '')
        plan_content = re.sub(r'\{"plan": "|"\}$', '', plan_content)

        console = Console()
        md = Markdown(plan_content)
        panel = Panel(md, title="OpenAI GPT-4 Output", border_style="blue")
        console.print(panel)

    except Exception as e:
        print(f"Error formatting output: {e}")
        print("Raw output:", output)
else:
    print("Please set your OpenAI API key to continue.")
